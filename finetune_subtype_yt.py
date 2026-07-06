"""finetune_subtype_yt.py — 차종 채널 파인튜닝 (AI-Hub + 유튜브 실채널 혼합)

근거(2026-07-03): 차종이 in-domain 89.2%인데 유튜브 실채널 1/5로 붕괴(채널 갭 — 폰 스피커
아닌 코덱/촬영채널 자체). 해법: 검증(한글 메타 + 검출게이트 + AI-Hub f0 지문 대조) 통과한
유튜브 사이렌 창을 AI-Hub와 혼합해 채널 강건성을 학습. 라벨은 영상 단위 약라벨.
사용자 라벨 5클립은 **held-out**(학습 미포함) — 채널 전이의 정직한 채점표.

  $ python finetune_subtype_yt.py --npz <ytft_train.npz> --heldout-dir <dir>
    (heldout wav 이름 규약: 클래스__이름.wav, 예: 경찰차__police1.wav)
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

import augment
import dataset as D
import infer
import models as M
from subtype_clf import SUBS, SUB_IDX, SubtypeDataset, siren_sub_chunks
from train import pick_device, seed_all


class YTMelDataset:
    """유튜브 사이렌 창(raw 멜 npz). 채널증강 → 창별 정규화 — SubtypeDataset과 동일 순서."""

    def __init__(self, npz, domain=True):
        d = np.load(npz, allow_pickle=True)
        self.X, self.Y = d["X"], d["Y"]
        self.domain = domain

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.domain:
            x = augment.domain_augment(x, np.random.default_rng())
        x = (x - x.mean()) / (x.std() + 1e-5)
        return torch.from_numpy(np.ascontiguousarray(x))[None], int(self.Y[i])


@torch.no_grad()
def eval_chunks(model, ds_eval, device):
    dl = DataLoader(ds_eval, batch_size=256)
    ok = n = 0
    for x, y in dl:
        p = model(x.to(device)).argmax(1).cpu()
        ok += int((p == y).sum()); n += len(y)
    return ok / max(1, n)


@torch.no_grad()
def eval_heldout(model, det, folder, device, tau=1.5):
    """클립 단위: 검출 게이트 창들의 차종 다수결 vs 파일명 라벨."""
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        cls = os.path.basename(f).split("__")[0]
        if cls not in SUB_IDX:
            continue
        y = D.load_wav(f)
        preds = []
        for _, m in infer.windows(y, 0.5):
            x = torch.from_numpy(np.ascontiguousarray(infer._norm(m)))[None, None].to(device)
            z = det(x)[0].cpu().numpy()
            if float(z[0] - max(z[1], z[2])) >= tau:
                preds.append(int(model(x).argmax(1)))
        maj = SUBS[Counter(preds).most_common(1)[0][0]] if preds else "-"
        rows.append((os.path.basename(f)[:-4], cls, maj, maj == cls, len(preds)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="ytft_train.npz (검증 통과 유튜브 창)")
    ap.add_argument("--heldout-dir", default=None, help="held-out wav 폴더 (클래스__이름.wav)")
    ap.add_argument("--base", default="models/subtype_cnn_attn_dom_s42.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--yt-frac", type=float, default=0.4, help="배치 내 유튜브 비율 기대값")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    seed_all(args.seed)
    device = pick_device()
    src = D.split_sources(D.index_sources())
    tr, va, te = (siren_sub_chunks(src, s) for s in ("train", "val", "test"))
    ai_ds = SubtypeDataset(tr, domain=True)                    # 채널증강 유지
    yt_ds = YTMelDataset(args.npz, domain=True)
    yt_y = yt_ds.Y
    print(f"AI-Hub train {len(ai_ds):,} / 유튜브 창 {len(yt_ds)} "
          f"({' '.join(f'{s}:{int((yt_y==i).sum())}' for i, s in enumerate(SUBS))}) / yt-frac {args.yt_frac}")

    # 혼합 가중치: 소스 비율(1-frac/frac) × 소스 내 클래스 균형
    cnt_ai = Counter(c.sub for c in tr)
    cnt_yt = Counter(int(v) for v in yt_y)
    w_ai = [(1 - args.yt_frac) / (3 * cnt_ai[c.sub]) for c in tr]
    w_yt = [args.yt_frac / (3 * cnt_yt[int(v)]) for v in yt_y]
    mixed = ConcatDataset([ai_ds, yt_ds])
    sampler = WeightedRandomSampler(w_ai + w_yt, num_samples=6000, replacement=True)
    dl = DataLoader(mixed, batch_size=64, sampler=sampler,
                    num_workers=args.workers, persistent_workers=args.workers > 0)

    model = M.build("cnn_attn").to(device)
    state = torch.load(args.base, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    det, _ = infer.load_model("models/cnn_attn_full_s42.pt", None, device)   # held-out 게이트용
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    va_ds = SubtypeDataset(va)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = n = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * len(y); n += len(y)
        model.eval()
        va_acc = eval_chunks(model, va_ds, device)
        print(f"  ep{ep:02d} loss {run/n:.3f}  AI-Hub val {va_acc*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    ckpt = f"models/subtype_cnn_attn_yt_s{args.seed}.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    print(f"[저장] {ckpt}")

    model.eval()
    te_acc = eval_chunks(model, SubtypeDataset(te), device)
    print(f"\nAI-Hub test: {te_acc*100:.1f}%  (기존 _dom 89.2% — 회귀 확인용)")
    if args.heldout_dir:
        print("held-out(학습 미포함 실채널 클립):")
        okc = nc = 0
        for name, cls, maj, ok, npred in eval_heldout(model, det, args.heldout_dir, device):
            print(f"  {name:24s} 정답 {cls}  예측 {maj:4s} {'✅' if ok else '❌'} ({npred}창)")
            okc += ok; nc += 1
        print(f"  → {okc}/{nc} (파인튜닝 전 1/5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
subtype_clf.py — 사이렌 차종 분별 측정 (HUD 차등 출력용, 통합모델 head 후보)

질문: 손수 특징 53%(docs/02)를 DL이 넘나? 경찰↔구급 겹침이 치명적인가?
  - siren 청크만 → subCategory {구급차/경찰차/소방차} 3-클래스
  - 라벨 이미 존재(합성 불필요), 차종 균형 샘플러 (구급차 ~2× 과다)
  - 증강 주의: 피치시프트는 차종이 피치에 일부 있어 라벨 흐림 → 기본 무증강으로 천장 측정

사용: python subtype_clf.py [--model cnn_attn] [--epochs 30] [--aug none|wave]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import augment
import dataset as D
import models as M
from train import confusion, prf, predict, seed_all, pick_device, lr_lambda_factory, _WorkerInit

SUBS = ["구급차", "경찰차", "소방차"]
SUB_IDX = {s: i for i, s in enumerate(SUBS)}


class SubtypeDataset:
    def __init__(self, chunks, transform=None, domain=False):
        self.chunks, self.transform, self.domain = chunks, transform, domain

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        c = self.chunks[i]
        x = self.transform(c) if self.transform is not None else D.chunk_mel(c)
        if self.domain:                                   # 채널(EQ·대역제한·잔향) 증강 — 정규화 전
            x = augment.domain_augment(x, np.random.default_rng())
        x = (x - x.mean()) / (x.std() + 1e-5)
        return torch.from_numpy(np.ascontiguousarray(x))[None], SUB_IDX[c.sub]


def siren_sub_chunks(src, split):
    return [c for c in D.build_chunks(src, classes=("siren",), splits=(split,))
            if c.sub in SUB_IDX]


def balanced_sampler(chunks):
    cnt = Counter(c.sub for c in chunks)
    w = [1.0 / cnt[c.sub] for c in chunks]
    return WeightedRandomSampler(w, num_samples=len(chunks), replacement=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_attn", choices=list(M.MODELS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", default="none", choices=["none", "wave"])
    ap.add_argument("--domain-aug", action="store_true",
                    help="채널(EQ·대역제한·잔향) 증강 — sim-to-real 강건성. _dom 체크포인트로 저장")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    seed_all(args.seed)
    device = pick_device()
    ckpt = f"models/subtype_{args.model}{'_dom' if args.domain_aug else ''}_s{args.seed}.pt"
    src = D.split_sources(D.index_sources())
    tr, va, te = (siren_sub_chunks(src, s) for s in ("train", "val", "test"))

    print(f"siren 차종 청크 — train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    for name, ch in (("train", tr), ("test", te)):
        c = Counter(x.sub for x in ch)
        print(f"  {name}: " + " ".join(f"{s} {c.get(s,0):,}" for s in SUBS))

    transform = None
    if args.aug != "none":
        from augment import MelAugment
        transform = MelAugment(args.aug, src, "train", seed=args.seed)

    dl_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = DataLoader(SubtypeDataset(tr, transform, domain=args.domain_aug), batch_size=64,
                          sampler=balanced_sampler(tr), worker_init_fn=_WorkerInit(args.seed), **dl_kw)
    val_dl = DataLoader(SubtypeDataset(va), batch_size=256, **dl_kw)
    test_dl = DataLoader(SubtypeDataset(te), batch_size=256, **dl_kw)

    model = M.build(args.model).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5 if args.model == "vit" else 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.epochs, warmup))

    best_f1, bad, t0 = -1.0, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        vp, vy = predict(model, val_dl, device)
        _, _, _, vf1 = prf(confusion(vy, vp.argmax(1)))
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if ep % 5 == 0 or bad >= 8:
            print(f"  ep{ep:02d} val macro-F1 {vf1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= 8:
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    tp, ty = predict(model, test_dl, device)
    cm = confusion(ty, tp.argmax(1))
    prec, rec, f1, macro = prf(cm)
    acc = np.trace(cm) / cm.sum()

    print(f"\n[{args.model} 차종] test 정확도 {acc*100:.1f}% · macro-F1 {macro:.3f} "
          f"(손수특징 53% 대비, 랜덤 33%)")
    print("혼동행렬 (행=정답, 열=예측):  " + "  ".join(SUBS))
    for i, s in enumerate(SUBS):
        print(f"  {s:5s} " + "  ".join(f"{cm[i,j]:5d}" for j in range(3)) +
              f"   recall {rec[i]:.2f}")
    print("\n경찰↔구급 혼동:", cm[0,1]+cm[1,0], "| 소방 분리도 recall:", round(rec[2],2))


if __name__ == "__main__":
    raise SystemExit(main())

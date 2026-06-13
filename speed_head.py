"""
speed_head.py — P2b: 통합모델 속도 head (방식 B/C) 학습 + 방식 A 대비

통합 모델: **검출 백본 동결** → 속도 head (docs/04 §축3, docs/06 §3.1).
  - 입력: 5초 윈도우 (검출·배포와 동일) — 방식 A의 12초 베이스라인보다 어려움(부분 전이)
  - 학습: synth_passby(v, d_min, SNR) on-the-fly, 라벨 = 우리가 정한 v (정확)
  - 손실: Huber(v̂, v).  v ~ U(0,80) 연속 (docs/06 §3.1)
  - 평가: test-split 정지 사이렌 합성 → **같은 5초 윈도우**에서 head MAE vs 방식 A MAE·기권율

⚠ 공정 비교 노트: speed_baseline.py(방식 A)는 12초·d=8m 유리조건(3–8 km/h)이었다.
   여기선 5초 윈도우(배포 현실, 부분 전이) → 둘 다 더 어려움. 동일 윈도우라 비교는 공정.

사용:
  python speed_head.py --backbone cnn_attn --ckpt models/cnn_attn_wave_spec_s42.pt
  python speed_head.py --backbone vit      --ckpt models/vit_wave_s42.pt        # 방식 C
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from torch.utils.data import DataLoader

import dataset as D
import doppler_speed as ds
import models as M
from train import pick_device, seed_all, lr_lambda_factory, _WorkerInit

SR = D.SR
WIN = 5 * SR                       # 5초 윈도우 (검출과 동일)
RESULTS = Path(__file__).resolve().parent / "results"


# ── 정지 베이스 (split 분리 — 누수 방지) ─────────────────────────────────
def stationary_pool(split, max_n, seed, glide_max=80.0):
    src = D.split_sources(D.index_sources())
    cand = [s for s in src if s.label == "siren" and s.acq == "인위적"
            and s.split == split and s.dur >= 6.0]
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    pool = []
    for s in cand:
        if len(pool) >= max_n:
            break
        try:
            sr, seg = ds.load_wav(s.wav, max_sec=12.0)
        except Exception:
            continue
        if len(seg) < sr * 6:
            continue
        _, f0 = ds.viterbi_f0(seg, sr)
        n = len(f0)
        if abs(np.median(f0[:n // 3]) - np.median(f0[-n // 3:])) <= glide_max:
            pool.append((seg.astype(np.float64), int(sr)))
    return pool


# ── 합성: 통과 → 5초 윈도우(중앙=최근접) → 22.05kHz ──────────────────────
def passby_window(seg, sr, v, d_min):
    pb = ds.synth_passby(seg, sr, v, d_min=d_min)
    g = np.gcd(sr, SR)
    pb = resample_poly(pb, SR // g, sr // g)
    c = len(pb) // 2                                  # 최근접 통과 = 중앙
    a = max(0, c - WIN // 2)
    w = pb[a:a + WIN]
    if len(w) < WIN:
        w = np.pad(w, (0, WIN - len(w)))
    return w.astype(np.float64)


def to_mel(w):
    m = D.logmel(w.astype(np.float32))
    if m.shape[1] < D.N_FRAMES:
        m = np.pad(m, ((0, 0), (0, D.N_FRAMES - m.shape[1])),
                   constant_values=np.float32(np.log(D.LOG_EPS)))
    x = m[:, :D.N_FRAMES]
    return (x - x.mean()) / (x.std() + 1e-5)


class SpeedSynth:
    """synth_passby on-the-fly. fixed=True면 인덱스 해시로 결정적(평가용)."""

    def __init__(self, pool, n, seed, fixed=False, snr_range=(5.0, 20.0)):
        self.pool, self.n, self.seed, self.fixed = pool, n, seed, fixed
        self.snr_range = snr_range

    def __len__(self):
        return self.n

    def _params(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i if self.fixed
                                    else np.random.randint(2 ** 31))
        seg, sr = self.pool[rng.integers(len(self.pool))]
        v = rng.uniform(0, 80)
        d = rng.uniform(5, 30)
        snr = rng.uniform(*self.snr_range)
        return seg, sr, v, d, snr, rng

    def __getitem__(self, i):
        seg, sr, v, d, snr, rng = self._params(i)
        w = passby_window(seg, sr, v, d)
        w = ds.add_noise(w, snr, rng)
        return torch.from_numpy(to_mel(w))[None].float(), np.float32(v)


# ── 속도 head ─────────────────────────────────────────────────────────────
class SpeedHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1))
        self.v = nn.Linear(64, 1)

    def forward(self, feat):
        return self.v(self.net(feat)).squeeze(-1)


def load_backbone(name, ckpt, device, freeze=True):
    m = M.build(name).to(device)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    for p in m.parameters():
        p.requires_grad_(not freeze)
    m.eval() if freeze else m.train()
    return m


# ── 방식 A vs head 평가 (동일 5초 윈도우) ────────────────────────────────
@torch.no_grad()
def evaluate(backbone, head, pool, device, snrs, d_mins, speeds, seed=999):
    rng = np.random.default_rng(seed)
    rows = []
    for snr in snrs:
        for d in d_mins:
            head_err, a_err, a_det, n = [], [], 0, 0
            for v in speeds:
                for seg, sr in pool:
                    w = passby_window(seg, sr, v, d)
                    w = ds.add_noise(w, snr, rng) if snr is not None else w
                    n += 1
                    # head
                    x = torch.from_numpy(to_mel(w))[None, None].float().to(device)
                    vh = float(head(backbone.features(x))[0])
                    head_err.append(abs(vh - v))
                    # 방식 A (같은 윈도우)
                    est = ds.estimate_passby(w, SR)
                    if est is not None:
                        a_det += 1
                        a_err.append(abs(est["v_kmh"] - v))
            rows.append({
                "snr": snr if snr is not None else "clean", "d_min": d, "n": n,
                "head_med": round(float(np.median(head_err)), 1),
                "head_mae": round(float(np.mean(head_err)), 1),
                "A_med": round(float(np.median(a_err)), 1) if a_err else None,
                "A_mae": round(float(np.mean(a_err)), 1) if a_err else None,
                "A_abstain": round(1 - a_det / n, 3),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="cnn_attn", choices=["cnn_attn", "vit"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=2000, help="epoch당 합성 샘플 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--finetune", action="store_true", help="백본 동결 해제 — 속도로 학습")
    ap.add_argument("--scratch", action="store_true", help="검출 ckpt 없이 from-scratch")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = pick_device()
    if args.smoke:
        args.epochs, args.steps, args.pool = 2, 200, 20

    print(f"정지 베이스 큐레이션 (train/test split 분리)…")
    tr_pool = stationary_pool("train", args.pool, args.seed)
    te_pool = stationary_pool("test", max(30, args.pool // 3), args.seed + 1)
    print(f"  train {len(tr_pool)} · test {len(te_pool)} 정지 클립")

    freeze = not (args.finetune or args.scratch)
    backbone = load_backbone(args.backbone, None if args.scratch else args.ckpt, device, freeze=freeze)
    head = SpeedHead().to(device)
    mode = "scratch" if args.scratch else ("finetune" if args.finetune else "frozen")
    if freeze:
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    else:                                   # 백본은 낮은 lr (전이 보존), head는 높게
        opt = torch.optim.AdamW([
            {"params": backbone.parameters(), "lr": 3e-4},
            {"params": head.parameters(), "lr": 1e-3},
        ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.epochs, 0))
    huber = nn.HuberLoss(delta=5.0)

    train_dl = DataLoader(SpeedSynth(tr_pool, args.steps, args.seed), batch_size=64,
                          num_workers=args.workers, worker_init_fn=_WorkerInit(args.seed),
                          persistent_workers=args.workers > 0)

    print(f"[{args.backbone} 속도 {mode}] device {device.type} · backbone "
          f"{'학습' if not freeze else '동결'}")
    t0 = time.time()
    for ep in range(args.epochs):
        head.train()
        if not freeze:
            backbone.train()
        run = 0.0
        for x, v in train_dl:
            x, v = x.to(device), v.to(device)
            if freeze:
                with torch.no_grad():
                    feat = backbone.features(x)
            else:
                feat = backbone.features(x)
            loss = huber(head(feat), v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * v.size(0)
        sched.step()
        print(f"  ep{ep:02d} huber {run/args.steps:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # 평가: 동일 5초 윈도우에서 head vs 방식 A
    head.eval()
    snrs = [None, 10.0, 5.0] if args.smoke else [None, 20.0, 10.0, 5.0]
    d_mins = [15.0] if args.smoke else [5.0, 15.0, 30.0]
    speeds = [30, 60] if args.smoke else [10, 30, 50, 70]
    rows = evaluate(backbone, head, te_pool, device, snrs, d_mins, speeds, seed=args.seed + 7)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"speed_head_{args.backbone}.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False))

    print(f"\n{'='*72}\n속도 오차 중앙값|평균 (km/h) — head({args.backbone}) vs 방식 A, **동일 5초 윈도우**\n{'='*72}")
    print(f"{'SNR':>6s} {'d_min':>6s} {'head 중앙|평균':>14s} {'방식A 중앙|평균':>15s} {'A기권':>6s}")
    for r in rows:
        a = f"{r['A_med']}|{r['A_mae']}" if r['A_med'] is not None else "기권"
        print(f"{str(r['snr']):>6s} {r['d_min']:>5.0f}m {r['head_med']:>6.1f}|{r['head_mae']:<6.1f} "
              f"{a:>14s} {r['A_abstain']*100:>5.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

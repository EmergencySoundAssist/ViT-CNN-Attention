"""
speed_seq.py — f0 궤적 시퀀스 속도 모델 (물리=추출, AI=매핑)

가설: 검출 백본 head가 속도에 실패한 건 "멜에서 글라이드 추출"이 어려워서다.
  → 추출은 물리(Viterbi f0 + envelope)에 맡기고, AI는 "글라이드 궤적 → 속도" 매핑만 학습.
  입력: 5초 윈도우의 [상대 log-f0(t), 정규화 envelope(t)] (2×L) — 저차원, 소형 bi-GRU.
  핵심: log-f0를 중앙값 제거(detrend)해 **절대 주파수(±56km/h 혼동원)를 빼고 글라이드 모양만** 남김
        = 물리의 source-무관 비율법과 같은 정보. AI는 그 매핑(+envelope 결합)을 학습.

평가: 동일 5초 윈도우에서 GRU vs 방식 A(물리) 중앙값 오차 비교.
정직한 천장: 클린 합성에선 물리와 비길 가능성 높음(둘 다 f0 최적 사용). 이득은 원거리·부분윈도우.

사용: python speed_seq.py [--train 4000] [--epochs 40] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import doppler_speed as ds
import speed_head as sh
from train import pick_device, seed_all, lr_lambda_factory

SR = sh.SR
L = 128                       # 고정 시퀀스 길이
RESULTS = Path(__file__).resolve().parent / "results"


def extract_features(w, sr=SR):
    """5초 윈도우 → (2, L): [중앙값 제거 log-f0, 정규화 envelope]."""
    _, f0 = ds.viterbi_f0(w, sr)
    _, env = ds.band_envelope(w, sr)
    logf = np.log(np.clip(np.nan_to_num(f0, nan=1.0), 1.0, None))
    logf = logf - np.median(logf)                 # 절대 주파수 제거 → 글라이드만
    env = env / (env.max() + 1e-9)
    rs = lambda x: np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(x)), x)
    return np.stack([rs(logf), rs(env)]).astype(np.float32)


class SeqSpeed(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(2, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):                          # x: (B, 2, L)
        o, _ = self.gru(x.transpose(1, 2))         # (B, L, 2H)
        return self.head(o.mean(dim=1)).squeeze(-1)


def gen(pool, n, seed, snr=None, v=None, d=None, keep_w=False):
    """synth_passby 윈도우 생성 → 특징·라벨(·윈도우). v/d/snr=None이면 무작위."""
    rng = np.random.default_rng(seed)
    X, Y, W = [], [], []
    for _ in range(n):
        seg, sr = pool[rng.integers(len(pool))]
        vv = rng.uniform(0, 80) if v is None else v
        dd = rng.uniform(5, 30) if d is None else d
        ss = rng.uniform(5, 20) if snr is None else snr
        w = sh.passby_window(seg, sr, vv, dd)
        if ss is not None:
            w = ds.add_noise(w, ss, rng)
        X.append(extract_features(w)); Y.append(np.float32(vv))
        if keep_w:
            W.append(w)
    return np.array(X), np.array(Y), W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train, args.epochs, args.pool = 300, 8, 25

    seed_all(args.seed)
    device = pick_device()
    print("정지 베이스 큐레이션…")
    tr_pool = sh.stationary_pool("train", args.pool, args.seed)
    te_pool = sh.stationary_pool("test", max(30, args.pool // 3), args.seed + 1)
    print(f"  train {len(tr_pool)} · test {len(te_pool)} 정지 클립")

    print(f"학습셋 합성·특징추출 {args.train}개 (Viterbi f0)…")
    t0 = time.time()
    Xtr, Ytr, _ = gen(tr_pool, args.train, args.seed)
    print(f"  완료 ({time.time()-t0:.0f}s), 특징 shape {Xtr.shape}")

    model = SeqSpeed().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.epochs, 0))
    huber = nn.HuberLoss(delta=5.0)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                    batch_size=64, shuffle=True)

    print(f"[f0-시퀀스 bi-GRU] 학습 · device {device.type} · params "
          f"{sum(p.numel() for p in model.parameters())/1e3:.0f}K")
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = huber(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * y.size(0)
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:02d} huber {run/len(Ytr):.2f} ({time.time()-t0:.0f}s)", flush=True)

    # 평가: 동일 5초 윈도우에서 GRU vs 물리
    model.eval()
    snrs = [None, 10.0, 5.0] if args.smoke else [None, 20.0, 10.0, 5.0]
    d_mins = [15.0] if args.smoke else [5.0, 15.0, 30.0]
    speeds = [30, 60] if args.smoke else [20, 50]
    rows = []
    for ss in snrs:
        for d in d_mins:
            gerr, aerr, adet, n = [], [], 0, 0
            for v in speeds:
                X, _, W = gen(te_pool, len(te_pool), args.seed + 700 + int(v), snr=ss, v=v, d=d, keep_w=True)
                with torch.no_grad():
                    vh = model(torch.from_numpy(X).to(device)).cpu().numpy()
                for k, w in enumerate(W):
                    n += 1
                    gerr.append(abs(float(vh[k]) - v))
                    est = ds.estimate_passby(w, SR)
                    if est is not None:
                        adet += 1; aerr.append(abs(est["v_kmh"] - v))
            rows.append({"snr": ss if ss is not None else "clean", "d_min": d, "n": n,
                         "gru_med": round(float(np.median(gerr)), 1),
                         "A_med": round(float(np.median(aerr)), 1) if aerr else None,
                         "A_abstain": round(1 - adet / n, 3)})

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "speed_seq.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"\n{'='*60}\n속도 오차 중앙값(km/h) — f0-시퀀스 GRU vs 방식 A(물리)\n{'='*60}")
    print(f"{'SNR':>6s} {'d_min':>6s} {'GRU':>7s} {'방식A':>7s} {'A기권':>6s}")
    for r in rows:
        a = f"{r['A_med']}" if r['A_med'] is not None else "기권"
        print(f"{str(r['snr']):>6s} {r['d_min']:>5.0f}m {r['gru_med']:>7.1f} {a:>7s} {r['A_abstain']*100:>5.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

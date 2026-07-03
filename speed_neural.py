"""
speed_neural.py — 순수 신경망 멜→속도 (물리 = 학습 시 f0 '선생', 런타임엔 없음)

비전: 출하 제품은 순수 학습 모델, 물리는 비계(scaffolding) → 업데이트 가능한 시스템.
방법:
  - synth_passby는 합성이라 **깨끗한 f0 정답을 공짜로** 만들 수 있다(노이즈 없는 합성에 Viterbi).
  - 입력 = 노이즈 낀 멜(64×216). 보조 타깃 = 깨끗한 합성의 f0 궤적(216). 주 타깃 = 속도 v.
  - conv 인코더(주파수만 풀링→시간축 보존) → ① f0 보조 head(글라이드 추출 강제) ② GRU 속도 head.
  - 손실 = Huber(v) + λ·MSE(f0). **런타임엔 물리 0** — 멜→conv→GRU→속도.

평가: 동일 5초 윈도우에서 신경망 속도 MAE vs 물리(방식 A), + 신경 f0가 Viterbi f0와 얼마나 맞나.

사용: python speed_neural.py [--train 4000] [--epochs 30] [--smoke]
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

from scipy.signal import resample_poly

import dataset as D
import doppler_speed as ds
import speed_head as sh
from train import pick_device, seed_all, lr_lambda_factory

SR = D.SR
L = D.N_FRAMES                        # 216 시간 프레임
WIN = 5 * SR                          # 5초 윈도우 샘플수
RESULTS = Path(__file__).resolve().parent / "results"

# 방향 클래스 (--dir-head): 창의 위상이 라벨의 근거. center(통과 전이) 창은 -1=CE 무시.
DIR_KO = ("정지", "접근", "멀어짐")
DIR_OF = {"still": 0, "approach": 1, "recede": 2, "center": -1}


def _logmel_hop(y, hop):
    """dataset.logmel과 동일하나 hop 가변 (시간해상도 조절). hop 256 → 2배 고운 프레임."""
    y = np.pad(y.astype(np.float32), D.N_FFT // 2, mode="reflect")
    n = 1 + (len(y) - D.N_FFT) // hop
    idx = np.arange(D.N_FFT)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * D._WINDOW, axis=1)) ** 2
    return np.log(spec @ D._FB.T + D.LOG_EPS).T.astype(np.float32)


def f0_target(w_clean, Ln):
    _, f0 = ds.viterbi_f0(w_clean.astype(np.float64), SR)
    lf = np.log(np.clip(np.nan_to_num(f0, nan=1.0), 1.0, None))
    lf = lf - np.median(lf)
    return np.interp(np.linspace(0, 1, Ln), np.linspace(0, 1, len(lf)), lf).astype(np.float32)


def mel_of(w, hop, Ln):
    m = _logmel_hop(w, hop)
    if m.shape[1] < Ln:
        m = np.pad(m, ((0, 0), (0, Ln - m.shape[1])), constant_values=np.float32(np.log(D.LOG_EPS)))
    m = m[:, :Ln]
    return ((m - m.mean()) / (m.std() + 1e-5)).astype(np.float32)


def mel_of_dom(w, hop, Ln, dom_rng):
    """mel_of + 채널(도메인) 증강 — 정규화 *전* EQ·대역제한·잔향. 학습 입력 강건화용."""
    import augment
    m = _logmel_hop(w, hop)
    if m.shape[1] < Ln:
        m = np.pad(m, ((0, 0), (0, Ln - m.shape[1])), constant_values=np.float32(np.log(D.LOG_EPS)))
    m = augment.domain_augment(m[:, :Ln], dom_rng)
    return ((m - m.mean()) / (m.std() + 1e-5)).astype(np.float32)


def _loop_to(seg, sr, sec, xfade=0.1):
    """seg를 crossfade로 이어붙여 최소 sec초 확보 — 사이렌=주기음이라 라벨 보존.
    접근/멀어짐 '분리 창'(통과 전이 안 걸침)을 5초로 뽑으려면 통과가 길어야 해서 필요."""
    need = int(sec * sr)
    if len(seg) >= need:
        return seg
    xf = max(1, int(xfade * sr))
    ramp = np.linspace(0.0, 1.0, xf)
    out = seg.astype(np.float64)
    while len(out) < need:
        head = seg[:xf] * ramp + out[-xf:] * (1.0 - ramp)
        out = np.concatenate([out[:-xf], head, seg[xf:]])
    return out[:need]


def window_of(seg, sr, v, d_min, phase, rng):
    """위상별 5초 클린 창(@SR). still=원본 정지 슬라이스(워프 없음, v=0의 진짜 분포),
    approach/recede=최근접에서 gap만큼 떨어진 순수 접근/멀어짐 창, center=기존(최근접=중앙)."""
    if phase == "still":
        a = int(rng.integers(0, max(1, len(seg) - int(5 * sr))))
        w = seg[a:a + int(5 * sr)]
        g = np.gcd(sr, SR)
        w = resample_poly(w, SR // g, sr // g)
    else:
        gap = int(0.7 * SR)
        seg2 = _loop_to(seg, sr, 13.0) if phase in ("approach", "recede") else seg
        pb = ds.synth_passby(seg2, sr, v, d_min=d_min)
        g = np.gcd(sr, SR)
        pb = resample_poly(pb, SR // g, sr // g)
        c = len(pb) // 2                               # 최근접 통과 = 중앙 (passby_window 관례)
        a = {"approach": c - gap - WIN, "recede": c + gap}.get(phase, c - WIN // 2)
        a = min(max(0, a), max(0, len(pb) - WIN))
        w = pb[a:a + WIN]
    if len(w) < WIN:
        w = np.pad(w, (0, WIN - len(w)))
    return w[:WIN].astype(np.float64)


def gen(pool, n, seed, hop, Ln, close=False, snr=None, v=None, d=None, keep_w=False,
        domain=False, dir_head=False, phase=None):
    """노이즈 멜(입력) + 깨끗한 f0(보조타깃) + v(주타깃) + 방향라벨 [+ 윈도우].
    close=True면 근거리 오버샘플. domain=True면 입력 멜 채널 증강(f0 타깃은 clean 유지).
    dir_head=True면 위상(approach/center/recede/still) 샘플링 → 방향 라벨 생성
    (phase로 고정 가능). False면 기존과 동일(center 창만, 방향 -1)."""
    rng = np.random.default_rng(seed)
    M, F, Y, DIR, W = [], [], [], [], []
    for _ in range(n):
        seg, sr = pool[rng.integers(len(pool))]
        ph = "center"
        if dir_head:
            ph = phase or str(rng.choice(["approach", "center", "recede", "still"],
                                         p=[0.3, 0.2, 0.3, 0.2]))
        vv = 0.0 if ph == "still" else (rng.uniform(0, 80) if v is None else v)
        if d is not None:
            dd = d
        elif close:                                    # 절반은 근거리 집중(3~12m), 절반은 전범위
            dd = rng.uniform(3, 12) if rng.random() < 0.5 else rng.uniform(5, 30)
        else:
            dd = rng.uniform(3, 30)                     # 균형: 근거리(3~)까지 포함, 쏠림 없음
        ss = rng.uniform(5, 20) if snr is None else snr
        w_clean = window_of(seg, sr, vv, dd, ph, rng)
        w = ds.add_noise(w_clean, ss, rng) if ss is not None else w_clean
        m = mel_of_dom(w, hop, Ln, rng) if domain else mel_of(w, hop, Ln)
        M.append(m); F.append(f0_target(w_clean, Ln)); Y.append(np.float32(vv))
        DIR.append(np.int64(DIR_OF[ph]))
        if keep_w:
            W.append(w)
    return np.array(M), np.array(F), np.array(Y), np.array(DIR), W


class NeuralSpeed(nn.Module):
    """멜 → (conv 인코더) → f0 보조 head + GRU {속도, 방향} head. 런타임 물리 없음.
    dir_head=True면 방향 3클래스(정지/접근/멀어짐) 로짓 추가 — 구 체크포인트와 하위호환."""

    def __init__(self, Ln, C=64, hidden=64, dir_head=False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, Ln)),                # 주파수만 1로, 시간 Ln 보존
        )
        self.f0head = nn.Conv1d(C, 1, 1)                  # 보조: 글라이드 추출 강제
        self.gru = nn.GRU(C, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.vhead = nn.Sequential(nn.Linear(2 * hidden, 64), nn.GELU(),
                                   nn.Dropout(0.1), nn.Linear(64, 1))
        self.dirhead = (nn.Sequential(nn.Linear(2 * hidden, 32), nn.GELU(), nn.Linear(32, 3))
                        if dir_head else None)

    def forward(self, x):                                 # x: (B,1,64,216)
        feat = self.enc(x).squeeze(2)                     # (B,C,L)
        f0 = self.f0head(feat).squeeze(1)                 # (B,L)
        o, _ = self.gru(feat.transpose(1, 2))             # (B,L,2H)
        pooled = o.mean(dim=1)
        v = self.vhead(pooled).squeeze(-1)
        if self.dirhead is None:
            return v, f0
        return v, f0, self.dirhead(pooled)                # dir: (B,3) 로짓


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--lam", type=float, default=5.0, help="f0 보조손실 가중")
    ap.add_argument("--fine", action="store_true", help="시간해상도 2배 (hop 256, L 432)")
    ap.add_argument("--close", action="store_true", help="근거리 오버샘플 학습")
    ap.add_argument("--domain-aug", action="store_true",
                    help="채널(EQ·대역제한·잔향) 증강 — sim-to-real 강건성. _dom 체크포인트로 저장")
    ap.add_argument("--dir-head", action="store_true",
                    help="방향 3클래스(정지/접근/멀어짐) 헤드 — 위상별 창 샘플링 + still(v=0) 포함. _dir 체크포인트")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train, args.epochs, args.pool = 300, 6, 25

    HOP = 256 if args.fine else D.HOP
    Ln = 1 + int(5.0 * SR) // HOP                     # 432(fine) / 216
    seed_all(args.seed)
    device = pick_device()
    print(f"설정: hop {HOP} · L {Ln} · 근거리오버샘플 {args.close} · fine {args.fine}")
    tr_pool = sh.stationary_pool("train", args.pool, args.seed)
    te_pool = sh.stationary_pool("test", max(30, args.pool // 3), args.seed + 1)
    print(f"  train {len(tr_pool)} · test {len(te_pool)} 정지 클립")

    print(f"학습셋 합성·멜·f0타깃 {args.train}개…" + (" (+방향 위상 샘플링)" if args.dir_head else ""))
    t0 = time.time()
    M, F, Y, DIRS, _ = gen(tr_pool, args.train, args.seed, HOP, Ln,
                           close=args.close, domain=args.domain_aug, dir_head=args.dir_head)
    print(f"  완료 ({time.time()-t0:.0f}s) 멜{M.shape} f0{F.shape}"
          + (f" 방향분포 {np.bincount(DIRS + 1, minlength=4).tolist()}(-1/정지/접근/멀어짐)" if args.dir_head else ""))

    model = NeuralSpeed(Ln, dir_head=args.dir_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.epochs, 0))
    huber, mse = nn.HuberLoss(delta=5.0), nn.MSELoss()
    ce = nn.CrossEntropyLoss(ignore_index=-1)         # center(전이) 창은 방향 라벨 무시
    dl = DataLoader(TensorDataset(torch.from_numpy(M)[:, None], torch.from_numpy(F),
                                  torch.from_numpy(Y), torch.from_numpy(DIRS)),
                    batch_size=64, shuffle=True)

    print(f"[순수 신경망 멜→속도{'+방향' if args.dir_head else ''}] params {sum(p.numel() for p in model.parameters())/1e3:.0f}K · {device.type}")
    for ep in range(args.epochs):
        model.train(); rv = rf = rd = 0.0
        for x, f, y, dd in dl:
            x, f, y, dd = x.to(device), f.to(device), y.to(device), dd.to(device)
            out = model(x)
            vh, f0 = out[0], out[1]
            lv, lf = huber(vh, y), mse(f0, f)
            loss = lv + args.lam * lf
            if args.dir_head:
                ld = ce(out[2], dd)
                loss = loss + ld
                rd += ld.item() * y.size(0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            rv += lv.item() * y.size(0); rf += lf.item() * y.size(0)
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            extra = f" dir-ce {rd/len(Y):.3f}" if args.dir_head else ""
            print(f"  ep{ep:02d} v-huber {rv/len(Y):.2f} f0-mse {rf/len(Y):.3f}{extra} ({time.time()-t0:.0f}s)", flush=True)

    # 체크포인트 저장 (infer.load_speed가 읽는 포맷: model/Ln/hop/fine[/dir])
    RESULTS.mkdir(exist_ok=True)
    ckpt = f"models/speed_neural{'_dir' if args.dir_head else ''}{'_dom' if args.domain_aug else ''}.pt"
    torch.save({"model": model.state_dict(), "Ln": Ln, "hop": HOP, "fine": args.fine,
                "dir": args.dir_head}, ckpt)
    print(f"[저장] {ckpt}")

    # 평가: 신경망 vs 물리 (동일 윈도우)
    model.eval()
    snrs = [None, 10.0, 5.0] if args.smoke else [None, 20.0, 10.0, 5.0]
    d_mins = [15.0] if args.smoke else [5.0, 15.0, 30.0]
    speeds = [30, 60] if args.smoke else [20, 50]
    rows = []
    for ss in snrs:
        for d in d_mins:
            gerr, aerr, adet, f0corr, n = [], [], 0, [], 0
            for v in speeds:
                Mx, Fx, _, _, W = gen(te_pool, len(te_pool), args.seed + 700 + int(v), HOP, Ln, snr=ss, v=v, d=d, keep_w=True)
                with torch.no_grad():
                    out = model(torch.from_numpy(Mx)[:, None].to(device))
                vh = out[0].cpu().numpy(); f0p = out[1].cpu().numpy()
                for k, w in enumerate(W):
                    n += 1; gerr.append(abs(float(vh[k]) - v))
                    c = np.corrcoef(f0p[k], Fx[k])[0, 1]
                    if np.isfinite(c):
                        f0corr.append(c)
                    est = ds.estimate_passby(w, SR)
                    if est is not None:
                        adet += 1; aerr.append(abs(est["v_kmh"] - v))
            rows.append({"snr": ss if ss is not None else "clean", "d_min": d,
                         "nn_med": round(float(np.median(gerr)), 1),
                         "A_med": round(float(np.median(aerr)), 1) if aerr else None,
                         "f0_corr": round(float(np.mean(f0corr)), 2) if f0corr else None,
                         "A_abstain": round(1 - adet / n, 3)})

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "speed_neural.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"\n{'='*66}\n순수 신경망 멜→속도 vs 물리 — 속도 중앙값오차(km/h), f0추출 상관\n{'='*66}")
    print(f"{'SNR':>6s} {'d_min':>6s} {'신경망':>7s} {'물리A':>7s} {'f0상관':>7s} {'A기권':>6s}")
    for r in rows:
        a = f"{r['A_med']}" if r['A_med'] is not None else "기권"
        fc = f"{r['f0_corr']}" if r['f0_corr'] is not None else "-"
        print(f"{str(r['snr']):>6s} {r['d_min']:>5.0f}m {r['nn_med']:>7.1f} {a:>7s} {fc:>7s} {r['A_abstain']*100:>5.0f}%")

    if args.dir_head:                                  # 방향 헤드 평가 (위상별, held-out 클립)
        n_eval = 60 if args.smoke else 150
        print(f"\n방향 헤드 평가 — 위상별 정확도·v̂중앙값 (SNR 10, d 3~30 랜덤, n={n_eval})")
        dstat = {}
        for ph in ("still", "approach", "recede"):
            Mx, _, _, Dx, _ = gen(te_pool, n_eval, args.seed + 990, HOP, Ln,
                                  snr=10.0, dir_head=True, phase=ph)
            with torch.no_grad():
                out = model(torch.from_numpy(Mx)[:, None].to(device))
            pred = out[2].argmax(1).cpu().numpy()
            acc = float((pred == Dx).mean())
            vmed = float(np.median(out[0].cpu().numpy()))
            dstat[ph] = {"acc": round(acc, 3), "v_med": round(vmed, 1)}
            print(f"  {DIR_KO[DIR_OF[ph]]:4s} acc {acc*100:5.1f}%   v̂중앙값 {vmed:5.1f} km/h")
        (RESULTS / "speed_dir_eval.json").write_text(json.dumps(dstat, ensure_ascii=False, indent=1))
        print("  (정지 v̂중앙값 ~0 근접 = OOD 바닥(6~10km/h) 완화 확인 지표)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

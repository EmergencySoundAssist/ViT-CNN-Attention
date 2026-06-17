"""
bench_speed.py — 물리 vs 신경망 속도, 동일 입력 A/B 벤치 (Mac 사니티 / Jetson 실측)

"나머지 다 동일, 추정기만 교체" — 같은 5초 윈도우를 물리(estimate_passby)와 신경망(NeuralSpeed)에
똑같이 넣어 ① 지연(latency) ② 정확도(합성 라벨) ③ A–B 일치율을 잰다.

지연 측정 규칙 (실제 파이프라인 반영):
  - 멜은 검출과 공유돼 한 번만 계산 → 신경망은 그 멜을 재사용(forward만 측정).
  - 물리(estimate_passby)는 자체 STFT를 내부에서 돌리므로 그 비용까지 포함(공유 불가).
  - GPU는 워밍업 + torch.cuda.synchronize()로 정확히 측정.

사용:
  python bench_speed.py                          # 지연만 (랜덤 가중치도 지연은 유효)
  python bench_speed.py --ckpt models/speed_neural.pt   # 지연 + 정확도 + 일치율
  python bench_speed.py --ckpt ... --fine        # 학습을 --fine으로 했으면 맞춰서
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import dataset as D
import doppler_speed as ds
import speed_head as sh
from speed_neural import NeuralSpeed, mel_of

SR = D.SR


def stat(xs):
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2]
    p95 = xs[min(n - 1, int(0.95 * n))]
    return med, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="NeuralSpeed 체크포인트 (없으면 지연만)")
    ap.add_argument("--n", type=int, default=120, help="벤치 윈도우 수")
    ap.add_argument("--pool", type=int, default=40, help="정지 베이스 클립 수")
    ap.add_argument("--fine", action="store_true", help="시간해상도 2배 (학습과 일치시킬 것)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    hop = 256 if args.fine else D.HOP
    Ln = 1 + int(5.0 * SR) // hop

    model = NeuralSpeed(Ln).to(dev).eval()
    trained = False
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        model.load_state_dict(ck["model"])
        trained = True
        if ck.get("fine") != args.fine:
            print(f"  ⚠ ckpt fine={ck.get('fine')} != --fine={args.fine} (해상도 불일치 주의)")
    print(f"device {dev.type} · 신경망 {'학습됨' if trained else '랜덤(지연만 유효)'} · L={Ln} · hop={hop}")

    pool = sh.stationary_pool("test", args.pool, args.seed + 1)
    if not pool:
        print("정지 베이스 0개 — 데이터(인위적 사이렌 wav) 확인 필요"); return 1

    rng = np.random.default_rng(args.seed)
    items = []                                       # (audio_window, mel, true_v)
    for _ in range(args.n):
        seg, sr = pool[rng.integers(len(pool))]
        v = rng.uniform(0, 80); d = rng.uniform(5, 30); snr = rng.uniform(5, 20)
        w = ds.add_noise(sh.passby_window(seg, sr, v, d), snr, rng)
        items.append((w, mel_of(w, hop, Ln), np.float32(v)))

    # 신경망 워밍업 (GPU 커널 컴파일/캐시)
    with torch.no_grad():
        x0 = torch.from_numpy(items[0][1])[None, None].to(dev)
        for _ in range(8):
            model(x0)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    lat_p, lat_n, err_p, err_n, agree = [], [], [], [], []
    ndet = 0
    for w, m, v in items:
        # 물리 (자체 STFT 포함)
        t0 = time.perf_counter()
        est = ds.estimate_passby(w, SR)
        lat_p.append((time.perf_counter() - t0) * 1000)
        vp = est["v_kmh"] if est is not None else None

        # 신경망 (공유 멜 재사용 → forward만)
        x = torch.from_numpy(m)[None, None].to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            vh, _ = model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        lat_n.append((time.perf_counter() - t0) * 1000)
        vn = float(vh[0])

        if trained:
            err_n.append(abs(vn - v))
            if vp is not None:
                ndet += 1; err_p.append(abs(vp - v)); agree.append(abs(vp - vn))

    mp, pp = stat(lat_p)
    mn, pn = stat(lat_n)
    print(f"\n{'='*60}\n물리 vs 신경망 — 동일 {args.n} 윈도우 (device {dev.type})\n{'='*60}")
    print(f"{'':12s} {'지연 중앙(ms)':>12s} {'p95(ms)':>9s}")
    print(f"{'물리 DSP':12s} {mp:>12.1f} {pp:>9.1f}   (자체 STFT 포함)")
    print(f"{'신경망':12s} {mn:>12.1f} {pn:>9.1f}   (공유 멜 재사용)")
    if trained:
        print(f"\n정확도·일치 (합성 라벨)")
        print(f"  물리   중앙오차 {np.median(err_p):.1f} km/h · 기권율 {(1-ndet/args.n)*100:.0f}%")
        print(f"  신경망 중앙오차 {np.median(err_n):.1f} km/h")
        print(f"  A–B 일치 중앙차 {np.median(agree):.1f} km/h")
    else:
        print("\n(정확도는 --ckpt 주면 측정 — 지금은 지연만)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

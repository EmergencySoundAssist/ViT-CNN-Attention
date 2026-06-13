"""
tdoa_sim.py — 마이크 어레이 TDOA 정확도 시뮬레이터 (하드웨어 이전 정량화)

목적: 4채널 1개 vs 8채널(양쪽 미러) 2개의 방향·거리 정확도를 측정 전에 추정.
synth_passby가 속도 베이스라인을 미리 잡은 것과 같은 수법 — 물리 합성 + 추정기 역산.

물리:
  - 합성: 구면파 (각 마이크까지 |source−mic|/c 지연 + 1/r 감쇠) — 근거리 시차(parallax) 포함
  - 추정: 마스크된 SRP-PHAT (사이렌 대역 700–1300 Hz 빈만, 평면파 가정) → bearing
  - 노이즈: 채널별 독립 백색(=확산/바람-like, 비상관 — TDOA의 hard case), 대역 내 SNR 기준
  - 거리: 두 어레이 bearing의 교점 (삼각측량) — 시차 한계를 실증

numpy + scipy. 실제 정지 사이렌 클립(speed_baseline 큐레이션)을 음원으로 사용.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.signal import resample_poly

import doppler_speed as ds
import speed_baseline as sb

C = 343.0
FS = 48000                       # 어레이 샘플레이트
BAND = (700.0, 1300.0)           # 사이렌 지배 대역 (마스킹)


# ── 기하 ──────────────────────────────────────────────────────────────────
def square_array(center, L=0.05):
    """4-mic 정사각 어레이 (변 L). center 기준 xy 좌표 (4,2)."""
    off = np.array([[-L/2, -L/2], [L/2, -L/2], [L/2, L/2], [-L/2, L/2]])
    return np.asarray(center, float)[None, :] + off


def pairs_of(m):
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


# ── 음원 로드 ─────────────────────────────────────────────────────────────
def load_siren(dur=0.5, seed=0):
    """정지 사이렌 1개 → FS로 리샘플, dur초 중앙 세그먼트."""
    pool = sb.curate_stationary(max_n=6, min_dur=4.0, seed=seed)
    sr, seg = pool[0][1], pool[0][2]
    g = np.gcd(int(sr), FS)
    s = resample_poly(seg, FS // g, int(sr) // g).astype(np.float64)
    n = int(dur * FS)
    a = max(0, len(s) // 2 - n // 2)
    return s[a:a + n]


# ── 합성 (구면파) ─────────────────────────────────────────────────────────
def simulate(mics, source_xy, sig, fs=FS):
    """각 마이크 신호 = sig를 |source−mic|/c 지연 + 1/r 감쇠 (FFT 위상 분수지연)."""
    n = len(sig)
    f = np.fft.rfftfreq(n, 1 / fs)
    Sf = np.fft.rfft(sig)
    out = np.zeros((len(mics), n))
    for i, p in enumerate(mics):
        r = float(np.hypot(*(source_xy - p)))
        Xi = Sf * np.exp(-1j * 2 * np.pi * f * (r / C)) / max(r, 0.1)
        out[i] = np.fft.irfft(Xi, n)
    return out


def _bandpower(x, band=BAND, fs=FS):
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X = np.fft.rfft(x)
    m = (f >= band[0]) & (f <= band[1])
    return np.sum(np.abs(X[m]) ** 2) / len(x)


def add_noise(ch, snr_db, rng, band=BAND, fs=FS):
    """채널별 독립 백색잡음, 대역 내 SNR=snr_db (확산/바람 모사)."""
    out = ch.copy()
    sp = np.mean([_bandpower(c) for c in ch])
    probe = rng.standard_normal(ch.shape[1])
    k = _bandpower(probe)                              # 단위분산 백색의 대역전력
    target = sp / 10 ** (snr_db / 10)
    for i in range(ch.shape[0]):
        out[i] += rng.standard_normal(ch.shape[1]) * np.sqrt(target / k)
    return out


# ── 추정: 마스크된 SRP-PHAT ───────────────────────────────────────────────
def srp_phat(ch, mics, az_grid, band=BAND, fs=FS):
    """대역 마스크 SRP-PHAT → (best_az, scores). 평면파 가정."""
    n = ch.shape[1]
    f = np.fft.rfftfreq(n, 1 / fs)
    bm = (f >= band[0]) & (f <= band[1])
    fb = f[bm]
    X = np.fft.rfft(ch, axis=1)[:, bm]                 # (M, Fb)
    prs = pairs_of(len(mics))
    Cij = []
    for i, j in prs:
        cs = X[i] * np.conj(X[j])
        Cij.append(cs / (np.abs(cs) + 1e-9))           # PHAT 백색화
    scores = np.empty(len(az_grid))
    for a, az in enumerate(az_grid):
        u = np.array([np.cos(az), np.sin(az)])
        s = 0.0
        for k, (i, j) in enumerate(prs):
            tau = -((mics[i] - mics[j]) @ u) / C
            s += np.real(np.sum(Cij[k] * np.exp(1j * 2 * np.pi * fb * tau)))
        scores[a] = s
    return az_grid[int(np.argmax(scores))], scores


def triangulate(c1, b1, c2, b2):
    """두 어레이 중심 c1,c2에서 bearing b1,b2 방향 광선의 교점."""
    u1 = np.array([np.cos(b1), np.sin(b1)])
    u2 = np.array([np.cos(b2), np.sin(b2)])
    A = np.column_stack([u1, -u2])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * u1


# ── 실험 ──────────────────────────────────────────────────────────────────
def ang_err_deg(est, true):
    d = (est - true + np.pi) % (2 * np.pi) - np.pi
    return abs(np.degrees(d))


def exp_bearing(sig, snrs, azis, L=0.05, R=40.0, reps=8, seed=42):
    """단일 4채널 어레이: bearing 오차 vs SNR."""
    rng = np.random.default_rng(seed)
    mics = square_array([0, 0], L)
    grid = np.deg2rad(np.arange(0, 360, 1.0))
    print(f"\n{'='*64}\n4채널 단일 어레이 — bearing 오차(°) vs SNR  (변 {L*100:.0f}cm, R={R:.0f}m)\n{'='*64}")
    print(f"{'SNR':>6s} " + " ".join(f"az{int(np.degrees(a)):>3d}°" for a in azis) + f" {'RMSE':>7s}")
    out = {}
    for snr in snrs:
        errs = []
        for az in azis:
            src = R * np.array([np.cos(az), np.sin(az)])
            e = []
            for _ in range(reps):
                ch = add_noise(simulate(mics, src, sig), snr, rng)
                est, _ = srp_phat(ch, mics, grid)
                e.append(ang_err_deg(est, az))
            errs.append(np.sqrt(np.mean(np.square(e))))
        rmse = np.sqrt(np.mean(np.square(errs)))
        out[snr] = rmse
        print(f"{snr:>5.0f}dB " + " ".join(f"{x:>5.1f}" for x in errs) + f" {rmse:>7.1f}")
    return out


def exp_distance(sig, ranges, snr, B=1.5, L=0.05, az=np.pi/2, reps=8, seed=7):
    """8채널(양쪽 미러 4ch×2): 거리 오차 vs 실제 거리 (삼각측량)."""
    rng = np.random.default_rng(seed)
    cL, cR = np.array([-B/2, 0.0]), np.array([B/2, 0.0])
    micsL, micsR = square_array(cL, L), square_array(cR, L)
    grid = np.deg2rad(np.arange(1, 180, 0.5))          # 전방 반평면
    print(f"\n{'='*64}\n8채널 양쪽어레이 — 거리 오차 vs 실제거리  (베이스라인 B={B}m, SNR={snr}dB)\n{'='*64}")
    print(f"{'실제R':>6s} {'추정R중앙':>9s} {'오차%':>7s} {'시차°(이론)':>11s} {'실패율':>7s}")
    for R in ranges:
        src = R * np.array([np.cos(az), np.sin(az)])
        ests, fails = [], 0
        for _ in range(reps):
            chL = add_noise(simulate(micsL, src, sig), snr, rng)
            chR = add_noise(simulate(micsR, src, sig), snr, rng)
            bL, _ = srp_phat(chL, micsL, grid)
            bR, _ = srp_phat(chR, micsR, grid)
            P = triangulate(cL, bL, cR, bR)
            if P is None or np.hypot(*P) > 10 * R or P[1] < 0:
                fails += 1
                continue
            ests.append(np.hypot(*P))
        parallax = np.degrees(B / R)                    # 근사 시차
        if ests:
            med = np.median(ests)
            print(f"{R:>5.0f}m {med:>8.1f}m {abs(med-R)/R*100:>6.0f}% "
                  f"{parallax:>10.2f}° {fails/reps*100:>6.0f}%")
        else:
            print(f"{R:>5.0f}m {'—':>8s}  {'실패':>6s} {parallax:>10.2f}° {fails/reps*100:>6.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sig = load_siren()
    print(f"음원: 정지 사이렌 {len(sig)/FS:.2f}s @ {FS}Hz, 대역전력 {_bandpower(sig):.2e}")

    # 사니티: 노이즈 없이 bearing 복원
    mics = square_array([0, 0])
    grid = np.deg2rad(np.arange(0, 360, 1.0))
    for true_deg in (30, 120, 200):
        az = np.deg2rad(true_deg)
        src = 40 * np.array([np.cos(az), np.sin(az)])
        est, _ = srp_phat(simulate(mics, src, sig), mics, grid)
        print(f"  사니티 az={true_deg}° → 추정 {np.degrees(est):.0f}° "
              f"(오차 {ang_err_deg(est, az):.1f}°)")

    if args.smoke:
        exp_bearing(sig, [10, 0], [np.deg2rad(45)], reps=3)
        return 0

    azis = [np.deg2rad(d) for d in (30, 90, 150)]
    exp_bearing(sig, [20, 10, 5, 0, -5, -10], azis, reps=10)
    exp_distance(sig, [10, 20, 30, 50, 100], snr=10, reps=12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

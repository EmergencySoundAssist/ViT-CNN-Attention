"""
Airacle 도플러 속도·방향 추정 모듈 (사이렌 전용, 견고형)

설계 근거 (데이터로 검증된 사실):
  - 도플러는 수신 파형 전체의 시간압축 → 피치 f(t)=f_src·k, 사이클 주기 T(t)=T_src/k (같은 인자 k)
  - 단일 클립 절대속도는 모집단 기준만으론 ±50km/h가 한계(개체 산포). 정밀값은 pass-by 자기참조.
  - pass-by 비율은 source-무관:  f_접근/f_이탈 = T_이탈/T_접근 = (c+v)/(c−v)
  - 견고화: 피치비와 사이클비(저주파, 잡음강인)를 둘 다 구해 평균

제공 기능:
  - pitch_track / cycle_period : sub-bin·sub-frame 보간된 정밀 추출
  - synth_passby               : 지연시간(retarded-time) 기반 물리적 pass-by 합성 (검증·데이터생성용)
  - estimate_passby            : 글라이드 검출 → 피치·사이클 비율 평균으로 v, 방향

numpy + scipy 만 사용.
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

C = 343.0          # 음속 m/s
NFFT = 2048
HOP = 256          # 11.6ms→ 프레임율 ~172Hz (사이클 sub-frame 정밀도용)
BAND = (300.0, 2500.0)


# ----------------------------- 입출력 -----------------------------
def load_wav(path, start=0.0, end=0.0, max_sec=12.0):
    """wav 로드. start/end는 **이 wav 내부** 초 단위 (AI Hub area 좌표 아님).

    AI Hub 클립 wav는 이미 annotation.area 구간만 잘린 파일이므로 (docs/01 함정 #3),
    전체 클립 분석은 기본값(start=0, end=0 → 처음부터 max_sec)을 쓴다.
    """
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / 32768.0
    a = int(max(0.0, start) * sr)
    b = int(min(end, start + max_sec) * sr) if end > start else int((start + max_sec) * sr)
    return sr, x[a:min(b, len(x))]


# --------------------------- 정밀 추출 ----------------------------
def _stft_mag(seg, sr):
    f, t, Z = stft(seg, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")
    return f, t, np.abs(Z)


def _parabolic(y_left, y_mid, y_right):
    """3점 포물선 정점 보정량 δ∈(-0.5,0.5)."""
    denom = y_left - 2 * y_mid + y_right
    return 0.5 * (y_left - y_right) / denom if denom != 0 else 0.0


def pitch_track(seg, sr, band=BAND, prom=2.5):
    """프레임별 지배 톤 주파수 트랙 f(t) (주파수 sub-bin 보간). 무효 프레임은 NaN."""
    f, t, mag = _stft_mag(seg, sr)
    m = (f >= band[0]) & (f <= band[1])
    fb, mb = f[m], mag[m]
    df = fb[1] - fb[0]
    med = np.median(mb, axis=0) + 1e-12
    track = np.full(mb.shape[1], np.nan)
    for j in range(mb.shape[1]):
        k = int(np.argmax(mb[:, j]))
        if mb[k, j] > prom * med[j]:
            d = _parabolic(mb[k - 1, j], mb[k, j], mb[k + 1, j]) if 0 < k < mb.shape[0] - 1 else 0.0
            track[j] = fb[k] + d * df
    return t, track


def band_envelope(seg, sr, band=BAND):
    f, t, mag = _stft_mag(seg, sr)
    m = (f >= band[0]) & (f <= band[1])
    return t, mag[m].sum(axis=0)


def cycle_period(sig, frame_rate, lag=(0.15, 6.0)):
    """자기상관 기반 사이클 주기(초) + 정규화 강도. sub-frame 포물선 보간."""
    s = np.asarray(sig, float)
    nan = np.isnan(s)
    if nan.all():
        return np.nan, 0.0
    if nan.any():
        idx = np.arange(s.size)
        s[nan] = np.interp(idx[nan], idx[~nan], s[~nan])
    s = s - s.mean()
    if np.allclose(s, 0):
        return np.nan, 0.0
    ac = np.correlate(s, s, "full")[s.size - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = max(1, int(lag[0] * frame_rate))
    hi = min(ac.size - 2, int(lag[1] * frame_rate))
    if hi <= lo + 1:
        return np.nan, 0.0
    seg = ac[lo:hi]
    cand = [(seg[i], lo + i) for i in range(1, seg.size - 1)
            if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]]
    k = max(cand)[1] if cand else lo + int(np.argmax(seg))
    d = _parabolic(ac[k - 1], ac[k], ac[k + 1]) if 0 < k < ac.size - 1 else 0.0
    return (k + d) / frame_rate, float(ac[k])


def _best_period(seg, sr):
    """주파수변조·진폭변조 중 더 강한 주기성을 사이클로 채택."""
    t, f0 = pitch_track(seg, sr)
    fr = 1.0 / (t[1] - t[0])
    p_f, s_f = cycle_period(f0, fr)
    _, env = band_envelope(seg, sr)
    p_e, s_e = cycle_period(env, fr)
    return (p_f, s_f) if s_f >= s_e else (p_e, s_e)


def _smooth(x, n):
    if n < 2:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


# -------------- 문맥 기반 피치 추적 (배음 널뛰기 제거) --------------
def salience_map(seg, sr, f0_lo=220.0, f0_hi=1400.0, n_f0=140, n_harm=5, stride=2):
    """프레임별 f0 salience (하모닉 합) — 진짜 기본주파수에서 피크."""
    f, t, mag = _stft_mag(seg, sr)
    mag = mag[:, ::stride]
    t = t[::stride]
    mag = mag / (mag.max(axis=0, keepdims=True) + 1e-9)
    df = f[1] - f[0]
    grid = np.exp(np.linspace(np.log(f0_lo), np.log(f0_hi), n_f0))
    sal = np.zeros((n_f0, mag.shape[1]))
    for h in range(1, n_harm + 1):
        idx = np.clip(np.round(grid * h / df).astype(int), 0, mag.shape[0] - 1)
        sal += mag[idx, :] / h                 # 낮은 배음에 가중
    return t, grid, sal


def viterbi_f0(seg, sr, lam=1.5):
    """
    시간연속성을 강제하는 Viterbi 피치 추적.
    프레임마다 독립적으로 최댓값을 고르지 않고(=배음 널뛰기 원인),
    주변 프레임과 매끄럽게 이어지는 최적 경로를 택함
    = '지금 사이클의 어느 위치인지'를 문맥으로 추론해 한 배음에 고정.
    lam: 클수록 점프 억제 강함.
    """
    t, grid, sal = salience_map(seg, sr)
    logS = np.log(sal + 1e-6)
    lg = np.log(grid)
    pen = lam * np.abs(lg[:, None] - lg[None, :])     # 로그주파수 점프 비용
    n, T = logS.shape
    dp = np.full((n, T), -1e18)
    back = np.zeros((n, T), int)
    dp[:, 0] = logS[:, 0]
    for j in range(1, T):
        prev = dp[:, j - 1][None, :] - pen
        back[:, j] = np.argmax(prev, axis=1)
        dp[:, j] = logS[:, j] + prev[np.arange(n), back[:, j]]
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(dp[:, -1]))
    for j in range(T - 1, 0, -1):
        path[j - 1] = back[path[j], j]
    return t, grid[path]


# ------------------------- pass-by 합성 ---------------------------
def synth_passby(seg, sr, v_kmh, d_min=8.0, attenuate=True):
    """
    정지 사이렌 클립을 물리적으로 올바른 통과(pass-by) 신호로 변환.
    차량이 속도 v로 직선도로(수직거리 d_min)를 지나며 관측자 앞을 통과.
    지연시간 t = τ + r(τ)/c 로 source시간 τ→관측시간 t 워핑 + 1/r 감쇠.
    """
    v = v_kmh / 3.6
    tau = np.arange(len(seg)) / sr
    t0 = tau[-1] / 2.0
    x = v * (tau - t0)                      # 차량 위치 (통과 시점 x=0)
    r = np.sqrt(d_min ** 2 + x ** 2)
    t_arr = tau + r / C                     # 관측자 도달 시각
    t_obs = np.linspace(t_arr[0], t_arr[-1], len(seg))
    tau_obs = np.interp(t_obs, t_arr, tau)  # 관측시간→source시간
    out = np.interp(tau_obs, tau, seg)
    if attenuate:
        r_obs = np.interp(tau_obs, tau, r)
        out = out * (d_min / np.maximum(r_obs, 1e-6))
    return out


def add_noise(seg, snr_db, rng=None):
    rng = rng or np.random.default_rng(0)
    n = rng.standard_normal(len(seg))
    sp = np.mean(seg ** 2) + 1e-12
    n *= np.sqrt(sp / (10 ** (snr_db / 10)) / (np.mean(n ** 2) + 1e-12))
    return seg + n


# ------------------------ 속도·방향 추정 --------------------------
def _avg_spectrum(seg, sr, band=BAND):
    f, _, mag = _stft_mag(seg, sr)
    m = (f >= band[0]) & (f <= band[1])
    return f[m], mag[m].mean(axis=1)


def _logspec(seg, sr, lo=350.0, hi=2200.0, n=600):
    """로그-주파수축으로 리샘플된 정규화 평균 스펙트럼."""
    f, S = _avg_spectrum(seg, sr)
    lf = np.linspace(np.log(lo), np.log(hi), n)
    Sl = np.interp(np.exp(lf), f, S)
    Sl = (Sl - Sl.mean()) / (Sl.std() + 1e-9)
    return lf, Sl


def doppler_factor(seg_app, seg_rec, sr, max_ratio=1.20):
    """
    접근·이탈 구간 평균 스펙트럼의 로그주파수 시프트로 (c+v)/(c−v) 추정.
    도플러는 모든 배음을 같은 비율로 옮기므로, 단일피크 추적과 달리
    배음 널뛰기·잡음에 강건(전 배음이 상호상관에 함께 기여).
    max_ratio: 물리적 상한(1.20≈상대속도 100km/h). 초과 lag는 배음 오정합 → 탐색 제외.
    반환: (ratio, sign, strength, prominence)
      strength   = 정규화 상호상관 피크값 (대략 corr 계수)
      prominence = 최고 피크가 차순위 국소피크보다 얼마나 우월한가 [0~1] (배음 모호성 게이트)
    """
    lf, Sa = _logspec(seg_app, sr)
    _, Sb = _logspec(seg_rec, sr)
    dlog = lf[1] - lf[0]
    cc = np.correlate(Sa, Sb, "full")
    center = Sa.size - 1
    maxlag = max(2, int(np.log(max_ratio) / dlog))
    win = cc[center - maxlag: center + maxlag + 1]
    k = int(np.argmax(win))
    # 차순위 국소 피크 (배음 오정합 시 비슷한 높이의 경쟁 피크가 존재)
    locs = [win[i] for i in range(1, win.size - 1)
            if win[i] > win[i - 1] and win[i] > win[i + 1] and abs(i - k) > 2]
    second = max(locs) if locs else 0.0
    prom = float(max(0.0, (win[k] - second) / (abs(win[k]) + 1e-9)))
    d = _parabolic(win[k - 1], win[k], win[k + 1]) if 0 < k < win.size - 1 else 0.0
    lag = (k - maxlag) + d
    return (float(np.exp(abs(lag * dlog))), float(np.sign(lag) or 1.0),
            float(win[k] / Sa.size), prom)


def estimate_passby(seg, sr, frac=0.3, min_strength=0.15):
    """
    통과(pass-by) 신호 → 속도(km/h)·방향.
      - 1차: 접근(앞 frac)·이탈(뒤 frac) 로그스펙트럼 상호상관 → v_spec (강건)
      - 교차검증: 진폭 envelope 사이클 주기 비율 → v_cycle (가능할 때 평균)
    글라이드가 약하면(등속/정지) None.
    """
    n = len(seg)
    app = seg[:int(frac * n)]
    rec = seg[int((1 - frac) * n):]
    ratio, sgn, strg, prom = doppler_factor(app, rec, sr)
    if strg < min_strength or prom < 0.12 or ratio <= 1.0:   # 약하거나 배음 모호 → 기권
        return None
    v_spec = C * (ratio - 1) / (ratio + 1)

    # 사이클(진폭 변조) 주기 비율 교차검증 — 빠른 yelp(경찰/구급)에서 유효
    ta, ea = band_envelope(app, sr)
    fr = 1.0 / (ta[1] - ta[0])
    _, eb = band_envelope(rec, sr)
    pa, sa = cycle_period(ea, fr)
    pb, sb = cycle_period(eb, fr)
    v_cycle = None
    if pa == pa and pb == pb and min(sa, sb) > 0.25 and pb / pa > 1.0:
        r_t = pb / pa
        v_cycle = C * (r_t - 1) / (r_t + 1)

    # 스펙트럼이 1차(강건). 사이클은 근접 시에만 평균(차이 작을 때만 신뢰).
    v = v_spec
    if v_cycle is not None and abs(v_cycle - v_spec) < 0.5 * v_spec:
        v = 0.5 * (v_spec + v_cycle)
    return {
        "v_kmh": v * 3.6,
        "v_spec": v_spec * 3.6,
        "v_cycle": (v_cycle * 3.6) if v_cycle is not None else None,
        "ratio": ratio,
        "strength": strg,
        "prominence": prom,
        "direction": "approach→recede" if sgn >= 0 else "recede→approach",
    }


# ---------------------------- 검증 -------------------------------
def _validate():
    import siren_data
    rng = np.random.default_rng(42)
    clips = siren_data.index_clips(classes=("siren",))
    # 정지 주기성이 강한 클립만 선별
    pool = []
    for c in clips:
        if len(pool) >= 25:
            break
        try:
            sr, seg = load_wav(c.wav)   # 클립 전체 (area 좌표 전달 금지)
        except Exception:
            continue
        if len(seg) < sr * 7:
            continue
        p, s = _best_period(seg, sr)
        if s > 0.35 and 0.15 < p < 5.5:
            pool.append((sr, seg))
    print(f"검증 클립: {len(pool)}개 (주기성 강한 사이렌)\n")

    speeds = [30, 50, 70]
    snrs = [("clean", None), ("20dB", 20), ("10dB", 10)]
    print(f"{'주입 v':>6s} | " + " | ".join(f"{n:>18s}" for n, _ in snrs))
    print("-" * 72)
    for v in speeds:
        cells = []
        for _, snr in snrs:
            errs, det = [], 0
            for sr, seg in pool:
                s2 = synth_passby(seg, sr, v)
                if snr is not None:
                    s2 = add_noise(s2, snr, rng)
                est = estimate_passby(s2, sr)
                if est is None:
                    continue
                det += 1
                errs.append(abs(est["v_kmh"] - v))
            if errs:
                cells.append(f"중앙값오차{np.median(errs):4.1f} 검출{det}/{len(pool)}")
            else:
                cells.append("검출 0")
        print(f"{v:>4d}km/h | " + " | ".join(f"{c:>20s}" for c in cells))

    # 스펙트럼 단독 vs 스펙트럼+사이클 평균 비교 (clean)
    print("\n[견고성] clean에서 추정기별 평균오차(km/h)")
    ep, eb, nc = [], [], 0
    for v in speeds:
        for sr, seg in pool:
            est = estimate_passby(synth_passby(seg, sr, v), sr)
            if est is None:
                continue
            ep.append(abs(est["v_spec"] - v))
            eb.append(abs(est["v_kmh"] - v))
            nc += est["v_cycle"] is not None
    print(f"   스펙트럼 단독       : {np.mean(ep):.1f}")
    print(f"   스펙트럼+사이클 평균 : {np.mean(eb):.1f}   (사이클 교차검증 적용 {nc}건)")


if __name__ == "__main__":
    _validate()

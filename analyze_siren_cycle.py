"""
사이렌 '피치 × 사이클 주기' 분리 가설 검증
- 가설: 차종(구급차/경찰차/소방차)은 기본주파수뿐 아니라 wail/yelp '사이클 주기'로도 갈린다
        예) "하이톤 + 느린 사이클 → 구급차"
- 측정: 클립별 (지배 톤 피치) + (주파수 변조 사이클 주기)
- 산출: 차종별 분포 + 2D 분리도 + 최근접중심(LOO) 분류 정확도 + scatter용 CSV

numpy + scipy 만 사용.
"""
import os, json, glob, random, csv, unicodedata
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

ROOT = "/Users/swlee/PycharmProjects/Airacle"
# AI Hub 트리 안 사이렌 디렉토리 (루트의 중복 추출본은 정리·삭제됨)
_T = "130.도시 소리 데이터/01.데이터"
JSON_DIRS = [f"{_T}/2.Validation/라벨링데이터/VL_1.교통소음/1.자동차/2.차량사이렌",
             f"{_T}/1.Training/라벨링데이터/TL_1.교통소음/1.자동차/2.차량사이렌"]
WAV_DIRS  = [f"{_T}/2.Validation/원천데이터/VS_1.교통소음/1.자동차/2.차량사이렌",
             f"{_T}/1.Training/원천데이터/TS_1.교통소음/1.자동차/2.차량사이렌"]
N_PER_CLASS = 200
REGION_SEC  = 12.0       # 여러 사이클을 담도록 길게
NFFT = 2048              # 46ms — 빠른 yelp도 잡히게 짧은 창
HOP  = 512               # 11.6ms → 프레임율 86Hz
BAND = (300.0, 2500.0)
CYCLE_LAG = (0.15, 6.0)  # 탐색할 사이클 주기 범위(초): 0.15s(빠른 yelp)~6s(느린 wail)
SEED = 42


def build_wav_index():
    idx = {}
    for d in WAV_DIRS:
        for p in glob.glob(os.path.join(ROOT, d, "*.wav")):
            idx[unicodedata.normalize("NFC", os.path.basename(p))] = p
    return idx


def load_region(path, start, end):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / 32768.0
    a = int(max(0, start) * sr)
    b = int(min(end, start + REGION_SEC) * sr) if end > start else int((start + REGION_SEC) * sr)
    return sr, x[a:min(b, len(x))]


def freq_track(sr, seg):
    """프레임별 지배 peak 주파수 트랙 + 대역 에너지 envelope, 프레임율 반환."""
    f, t, Z = stft(seg, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")
    mag = np.abs(Z)
    band = (f >= BAND[0]) & (f <= BAND[1])
    fb, mb = f[band], mag[band, :]
    if mb.shape[1] < 8:
        return None, None, None
    med = np.median(mb, axis=0) + 1e-12
    track = np.full(mb.shape[1], np.nan)
    for j in range(mb.shape[1]):
        k = int(np.argmax(mb[:, j]))
        if mb[k, j] > 2.5 * med[j]:
            track[j] = fb[k]
    env = mb.sum(axis=0)
    frame_rate = sr / HOP
    return track, env, frame_rate


def dominant_pitch(track):
    v = track[~np.isnan(track)]
    return float(np.median(v)) if v.size else np.nan


def cycle_period(sig, frame_rate):
    """자기상관으로 사이클 주기(초) 추정. (period, 정규화peak강도) 반환."""
    s = np.asarray(sig, float)
    nan = np.isnan(s)
    if nan.all():
        return np.nan, 0.0
    if nan.any():  # 결측 선형보간
        idx = np.arange(s.size)
        s[nan] = np.interp(idx[nan], idx[~nan], s[~nan])
    s = s - s.mean()
    if np.allclose(s, 0):
        return np.nan, 0.0
    ac = np.correlate(s, s, "full")[s.size - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = max(1, int(CYCLE_LAG[0] * frame_rate))
    hi = min(ac.size - 1, int(CYCLE_LAG[1] * frame_rate))
    if hi <= lo + 1:
        return np.nan, 0.0
    seg = ac[lo:hi]
    # 국소 최대 중 가장 강한 것
    peaks = [(seg[i], lo + i) for i in range(1, seg.size - 1)
             if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]]
    if not peaks:
        k = lo + int(np.argmax(seg)); return k / frame_rate, float(ac[k])
    strength, lag = max(peaks)
    return lag / frame_rate, float(strength)


def main():
    random.seed(SEED)
    wav_idx = build_wav_index()

    recs = []
    for d in JSON_DIRS:
        for jp in glob.glob(os.path.join(ROOT, d, "*.json")):
            try:
                m = json.load(open(jp, encoding="utf-8"))
            except Exception:
                continue
            ann = (m.get("annotations") or [{}])[0]
            ln = ann.get("labelName")
            wp = wav_idx.get(unicodedata.normalize("NFC", ln)) if ln else None
            if not wp:
                continue
            recs.append({"wav": wp, "sub": ann.get("subCategory", "?"),
                         "start": float(ann.get("area", {}).get("start", 0) or 0),
                         "end": float(ann.get("area", {}).get("end", 0) or 0)})

    by_sub = {}
    for r in recs:
        by_sub.setdefault(r["sub"], []).append(r)
    sampled = []
    for sub, rs in by_sub.items():
        random.shuffle(rs); sampled.extend(rs[:N_PER_CLASS])
    print(f"표본 {len(sampled)}건 분석 (차종별 최대 {N_PER_CLASS})\n")

    rows = []
    for i, r in enumerate(sampled):
        try:
            sr, seg = load_region(r["wav"], r["start"], r["end"])
            if len(seg) < sr * 1.0:
                continue
            track, env, fr = freq_track(sr, seg)
            if track is None or np.isnan(track).mean() > 0.6:
                continue
            pitch = dominant_pitch(track)
            p_f, s_f = cycle_period(track, fr)   # 주파수 변조 사이클
            p_e, s_e = cycle_period(env, fr)     # 진폭 변조 사이클
            period, strg = (p_f, s_f) if s_f >= s_e else (p_e, s_e)
            if not (period == period) or strg < 0.2:   # 주기성 약하면 제외
                continue
            rows.append({"sub": r["sub"], "pitch": pitch,
                         "period": period, "rate": 1.0 / period, "strength": strg})
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(sampled)}")

    print(f"\n주기성 유효 분석: {len(rows)}건\n")
    subs = sorted(set(r["sub"] for r in rows))

    def stat(vals):
        v = np.array(vals)
        return f"중앙값 {np.median(v):6.2f}  IQR[{np.percentile(v,25):.2f}–{np.percentile(v,75):.2f}]"

    print("=" * 76)
    print("[차종별] 피치(Hz) · 사이클 주기(s) · 사이클 속도(Hz)")
    print("=" * 76)
    for sub in subs:
        rs = [r for r in rows if r["sub"] == sub]
        print(f"\n● {sub}  (n={len(rs)})")
        print(f"   피치  : {stat([r['pitch']  for r in rs])} Hz")
        print(f"   주기  : {stat([r['period'] for r in rs])} s   (작을수록 빠른 사이클)")
        print(f"   속도  : {stat([r['rate']   for r in rs])} Hz")

    # 2D 분리도: 최근접 중심(leave-one-out) 분류 정확도
    feats = np.array([[r["pitch"], np.log(r["period"])] for r in rows])
    labels = np.array([subs.index(r["sub"]) for r in rows])
    mu, sd = feats.mean(0), feats.std(0) + 1e-9
    Z = (feats - mu) / sd
    centroids = {c: Z[labels == c].mean(0) for c in range(len(subs))}
    correct = 0
    conf = np.zeros((len(subs), len(subs)), int)
    for i in range(len(Z)):
        # i 제외한 중심
        best, bd = -1, 1e9
        for c in range(len(subs)):
            mask = (labels == c); mask[i] = False
            cen = Z[mask].mean(0)
            d = np.sum((Z[i] - cen) ** 2)
            if d < bd:
                bd, best = d, c
        conf[labels[i], best] += 1
        correct += (best == labels[i])
    acc = correct / len(Z)
    print("\n" + "=" * 76)
    print(f"[피치×사이클 2D] 최근접중심 LOO 분류 정확도: {acc*100:.1f}%  (3클래스 랜덤=33%)")
    print("  혼동행렬 (행=정답, 열=예측):  " + "  ".join(subs))
    for i, sub in enumerate(subs):
        print(f"    {sub:6s} " + "  ".join(f"{conf[i,j]:4d}" for j in range(len(subs))))

    # scatter용 CSV
    csv_path = os.path.join(ROOT, "siren_cycle_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sub", "pitch", "period", "rate", "strength"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"\nCSV 저장: {csv_path}")


if __name__ == "__main__":
    main()

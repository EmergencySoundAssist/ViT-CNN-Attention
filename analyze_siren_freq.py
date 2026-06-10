"""
한국 사이렌 정지 기준 주파수(f_src) 실측
- AI Hub 차량사이렌 클립에서 지배적 톤 주파수와 sweep 범위를 추출
- 차종(구급차/경찰차/소방차) · 취득방식(인위적/자연적)별 분포 집계
- 목적: f_src가 좁게 군집되면 단일 관측 도플러 속도 추정이 가능해짐

numpy + scipy 만 사용. librosa 불필요.
"""
import os, json, glob, random, csv, unicodedata
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

ROOT = "/Users/swlee/PycharmProjects/Airacle"
# AI Hub 트리 안 사이렌 디렉토리 (루트의 중복 추출본은 정리·삭제됨)
_T = "130.도시 소리 데이터/01.데이터"
JSON_DIRS = [f"{_T}/2.Validation/라벨링데이터/VL_1.교통소음/1.자동차/2.차량사이렌",
             f"{_T}/1.Training/라벨링데이터/TL_1.교통소음/1.자동차/2.차량사이렌"]   # val, train 라벨
WAV_DIRS  = [f"{_T}/2.Validation/원천데이터/VS_1.교통소음/1.자동차/2.차량사이렌",
             f"{_T}/1.Training/원천데이터/TS_1.교통소음/1.자동차/2.차량사이렌"]    # val, train wav
N_PER_CLASS = 220        # 차종별 최대 표본 수
REGION_SEC  = 8.0        # 클립당 분석 구간 길이(초)
NFFT = 8192
HOP  = 4096
BAND = (300.0, 2500.0)   # 사이렌 톤 탐색 대역
F0_BAND = (250.0, 1600.0)  # HPS 기본주파수 탐색 대역
SEED = 42


def build_wav_index():
    idx = {}
    for d in WAV_DIRS:
        for p in glob.glob(os.path.join(ROOT, d, "*.wav")):
            # macOS 파일시스템은 한글을 NFD로 저장 → JSON(NFC)과 맞추려면 정규화 필수
            idx[unicodedata.normalize("NFC", os.path.basename(p))] = p
    return idx


def load_region(path, start, end):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if np.issubdtype(np.int16, np.integer):
        x /= 32768.0
    a = int(max(0, start) * sr)
    b = int(min(end, start + REGION_SEC) * sr) if end > start else int((start + REGION_SEC) * sr)
    b = min(b, len(x))
    seg = x[a:b]
    return sr, seg


def dominant_track(sr, seg):
    """프레임별 대역 내 지배적 peak 주파수(prominence 필터)."""
    f, t, Z = stft(seg, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")
    mag = np.abs(Z)
    band = (f >= BAND[0]) & (f <= BAND[1])
    fb = f[band]
    mb = mag[band, :]
    if mb.shape[1] == 0 or fb.size == 0:
        return np.array([]), f, mag
    peaks = []
    med = np.median(mb, axis=0) + 1e-12
    for j in range(mb.shape[1]):
        col = mb[:, j]
        k = int(np.argmax(col))
        if col[k] > 3.0 * med[j]:        # 톤이 충분히 두드러진 프레임만
            peaks.append(fb[k])
    return np.array(peaks), f, mag


def hps_f0(f, mag):
    """Harmonic Product Spectrum 으로 기본주파수 추정(시간평균 스펙트럼)."""
    avg = mag.mean(axis=1)
    df = f[1] - f[0]
    hps = avg.copy()
    for h in (2, 3, 4):
        dec = avg[::h]
        hps[:len(dec)] *= dec
    lo = int(F0_BAND[0] / df); hi = int(F0_BAND[1] / df)
    lo = max(lo, 1); hi = min(hi, len(hps))
    if hi <= lo:
        return float("nan")
    k = lo + int(np.argmax(hps[lo:hi]))
    return float(k * df)


def main():
    random.seed(SEED)
    wav_idx = build_wav_index()
    print(f"WAV 인덱스: {len(wav_idx)}개")

    # JSON 수집 + 차종별 표본 추출
    recs = []
    for d in JSON_DIRS:
        for jp in glob.glob(os.path.join(ROOT, d, "*.json")):
            try:
                m = json.load(open(jp, encoding="utf-8"))
            except Exception:
                continue
            ann = (m.get("annotations") or [{}])[0]
            env = m.get("environment", {})
            label_name = ann.get("labelName")
            if label_name:
                label_name = unicodedata.normalize("NFC", label_name)
            wp = wav_idx.get(label_name) if label_name else None
            if not wp:
                continue
            recs.append({
                "wav": wp,
                "sub": ann.get("subCategory", "?"),
                "acq": env.get("acqMethod", "?"),
                "dist": env.get("distance", "?"),
                "start": float(ann.get("area", {}).get("start", 0) or 0),
                "end": float(ann.get("area", {}).get("end", 0) or 0),
            })

    by_sub = {}
    for r in recs:
        by_sub.setdefault(r["sub"], []).append(r)
    sampled = []
    for sub, rs in by_sub.items():
        random.shuffle(rs)
        sampled.extend(rs[:N_PER_CLASS])
    print(f"전체 매칭 {len(recs)}건 → 표본 {len(sampled)}건 분석\n")

    out_rows = []
    for i, r in enumerate(sampled):
        try:
            sr, seg = load_region(r["wav"], r["start"], r["end"])
            if len(seg) < sr * 0.5:
                continue
            peaks, f, mag = dominant_track(sr, seg)
            if peaks.size < 3:
                continue
            dom = float(np.median(peaks))
            p10, p90 = float(np.percentile(peaks, 10)), float(np.percentile(peaks, 90))
            f0 = hps_f0(f, mag)
            out_rows.append({**r, "dom": dom, "p10": p10, "p90": p90, "f0": f0,
                             "sweep": p90 - p10, "n_frames": int(peaks.size)})
        except Exception as e:
            continue
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(sampled)} 처리")

    print(f"\n유효 분석: {len(out_rows)}건\n")

    # CSV 저장
    csv_path = os.path.join(ROOT, "siren_freq_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["wav", "sub", "acq", "dist",
                                             "dom", "p10", "p90", "sweep", "f0", "n_frames"])
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k) for k in w.fieldnames})

    def stats(vals):
        v = np.array([x for x in vals if x == x])  # drop nan
        if v.size == 0:
            return "n=0"
        return (f"n={v.size:3d}  median={np.median(v):6.0f}Hz  "
                f"IQR=[{np.percentile(v,25):.0f}–{np.percentile(v,75):.0f}]  "
                f"p5–p95=[{np.percentile(v,5):.0f}–{np.percentile(v,95):.0f}]")

    print("=" * 78)
    print("[차종별] 지배적 톤 주파수 (dom = 프레임별 peak의 중앙값)")
    print("=" * 78)
    for sub in sorted(set(r["sub"] for r in out_rows)):
        rows = [r for r in out_rows if r["sub"] == sub]
        print(f"\n● {sub}")
        print(f"   dom   : {stats([r['dom'] for r in rows])}")
        print(f"   f0(HPS): {stats([r['f0'] for r in rows])}")
        print(f"   sweep : {stats([r['sweep'] for r in rows])}  (클립 내 p90-p10, wail/yelp 폭)")

    print("\n" + "=" * 78)
    print("[취득방식별] dom 분포 — 자연적(실환경=이동가능) vs 인위적(제작=정지)")
    print("=" * 78)
    for acq in sorted(set(r["acq"] for r in out_rows)):
        rows = [r for r in out_rows if r["acq"] == acq]
        print(f"   {acq:6s}: {stats([r['dom'] for r in rows])}")

    # 군집 히스토그램 (dom)
    print("\n" + "=" * 78)
    print("[전체 dom 히스토그램] 100Hz bin")
    print("=" * 78)
    doms = np.array([r["dom"] for r in out_rows])
    edges = np.arange(300, 2600, 100)
    hist, _ = np.histogram(doms, bins=edges)
    mx = max(hist.max(), 1)
    for c, h in zip(edges[:-1], hist):
        bar = "█" * int(40 * h / mx)
        if h:
            print(f"  {c:4d}-{c+100:<4d}Hz | {h:3d} {bar}")

    print(f"\nCSV 저장: {csv_path}")


if __name__ == "__main__":
    main()

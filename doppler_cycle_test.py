"""
'사이클 주기로 속도 추정' 가설 검증
1) 물리: 배속 k(=등속 도플러)를 가하면 피치 ×k, 사이클 주기 ×(1/k) 인가?
2) 절대속도 앵커 정밀도:
   - 모집단 기준(전 경찰 클립의 정지 사이클 중앙값)만으로 속도를 얼마나 맞히나
   - 자기참조(그 클립 자신의 정지 사이클)로는 얼마나 맞히나
3) 비교용으로 피치 기준 앵커의 정밀도도 같이

numpy + scipy. analyze_siren_cycle 의 함수 재사용.
"""
import os, json, glob, random, csv, unicodedata
import numpy as np
from analyze_siren_cycle import (build_wav_index, load_region, freq_track,
                                 cycle_period, dominant_pitch, ROOT, JSON_DIRS)

C = 343.0  # 음속 m/s
SEED = 7


def resample_speed(seg, v_kmh):
    """접근속도 v_kmh 의 등속 도플러 = 배속 k=c/(c-v) 리샘플."""
    v = v_kmh / 3.6
    k = C / (C - v)
    n = len(seg)
    new_len = max(8, int(n / k))
    idx = np.linspace(0, n - 1, new_len)
    return np.interp(idx, np.arange(n), seg)


def measure(sr, seg):
    track, env, fr = freq_track(sr, seg)
    if track is None or np.isnan(track).mean() > 0.6:
        return None, None
    pitch = dominant_pitch(track)
    p_f, s_f = cycle_period(track, fr)
    p_e, s_e = cycle_period(env, fr)
    period = p_f if s_f >= s_e else p_e
    strg = max(s_f, s_e)
    if not (period == period) or strg < 0.25:
        return pitch, None
    return pitch, period


def main():
    random.seed(SEED)
    wav_idx = build_wav_index()

    # 경찰차 클립 수집 (사이클이 가장 표준화돼 있음)
    police = []
    for d in JSON_DIRS:
        for jp in glob.glob(os.path.join(ROOT, d, "*.json")):
            try:
                m = json.load(open(jp, encoding="utf-8"))
            except Exception:
                continue
            ann = (m.get("annotations") or [{}])[0]
            if ann.get("subCategory") != "경찰차":
                continue
            ln = ann.get("labelName")
            wp = wav_idx.get(unicodedata.normalize("NFC", ln)) if ln else None
            if wp:
                police.append({"wav": wp})
    random.shuffle(police)

    # ---------- (A) 물리 검증 + 속도 복원 ----------
    test_speeds = [-40, -20, 20, 40]   # 접근(+)/이탈(-) km/h
    phys_err_T, phys_err_f = [], []
    rec_self, rec_pop, rec_pitch = {v: [] for v in test_speeds}, {v: [] for v in test_speeds}, {v: [] for v in test_speeds}

    base_T, base_f = [], []
    n_used = 0
    for r in police:
        if n_used >= 30:
            break
        try:
            sr, seg = load_region(r["wav"])   # 클립 전체 (area 좌표 전달 금지)
        except Exception:
            continue
        if len(seg) < sr * 2:
            continue
        f0, T0 = measure(sr, seg)
        if T0 is None or not (0.28 < T0 < 0.40):  # 표준 yelp 군집만
            continue
        base_T.append(T0); base_f.append(f0)
        n_used += 1
        for v in test_speeds:
            k = C / (C - v / 3.6)
            seg_v = resample_speed(seg, v)
            f_v, T_v = measure(sr, seg_v)
            if T_v is None:
                continue
            # 물리: T_v ≈ T0/k, f_v ≈ f0*k
            phys_err_T.append(abs(T_v - T0 / k) / (T0 / k) * 100)
            phys_err_f.append(abs(f_v - f0 * k) / (f0 * k) * 100)
            # 속도 복원: v = c(1 - T_obs/T_ref)
            rec_self[v].append(C * (1 - T_v / T0) * 3.6)            # 자기참조

    pop_T = float(np.median(base_T))   # 모집단 정지 사이클 기준
    pop_f = float(np.median(base_f))
    # 모집단/피치 기준 복원은 base 수집 후 다시 계산
    for r in police[:80]:
        try:
            sr, seg = load_region(r["wav"])
        except Exception:
            continue
        if len(seg) < sr * 2:
            continue
        f0, T0 = measure(sr, seg)
        if T0 is None or not (0.28 < T0 < 0.40):
            continue
        for v in test_speeds:
            k = C / (C - v / 3.6)
            f_v, T_v = measure(sr, resample_speed(seg, v))
            if T_v is None:
                continue
            rec_pop[v].append(C * (1 - T_v / pop_T) * 3.6)     # 사이클 모집단기준
            rec_pitch[v].append(C * (1 - pop_f / f_v) * 3.6)   # 피치 모집단기준

    print("=" * 72)
    print(f"분석 경찰 클립: {n_used}개,  정지 사이클 중앙값 T_src={pop_T:.3f}s,  피치 중앙값 f_src={pop_f:.0f}Hz")
    print("=" * 72)
    print(f"\n[물리 검증] 배속 k 적용 시 실제 변화가 예측과 얼마나 맞나 (오차%)")
    print(f"   사이클 T_obs vs T0/k : 평균오차 {np.mean(phys_err_T):.1f}%")
    print(f"   피치   f_obs vs f0*k : 평균오차 {np.mean(phys_err_f):.1f}%")

    print(f"\n[절대속도 복원 오차] 주입속도 대비 (RMSE km/h)")
    print(f"  {'주입 v':>8s} | {'사이클-자기참조':>14s} | {'사이클-모집단기준':>16s} | {'피치-모집단기준':>14s}")
    def rmse(arr, v):
        a = np.array(arr)
        return np.sqrt(np.mean((a - v) ** 2)) if a.size else float('nan')
    for v in test_speeds:
        print(f"  {v:>6d}   |  {rmse(rec_self[v], v):>12.1f}  |   {rmse(rec_pop[v], v):>13.1f}  |  {rmse(rec_pitch[v], v):>12.1f}")

    # ---------- (B) 모집단 산포 → 절대속도 하한 ----------
    print("\n" + "=" * 72)
    print("[모집단 기준의 한계] 정지 기준값 자체의 산포가 절대속도 정밀도의 바닥")
    print("=" * 72)
    rows = list(csv.DictReader(open(os.path.join(ROOT, "siren_cycle_stats.csv"), encoding="utf-8")))
    for sub in ["경찰차", "구급차", "소방차"]:
        T = np.array([float(x["period"]) for x in rows if x["sub"] == sub and 0.28 < float(x["period"]) < 0.40])
        F = np.array([float(x["pitch"]) for x in rows if x["sub"] == sub])
        if T.size:
            cvT = T.std() / T.mean()
            cvF = F.std() / F.mean()
            print(f"  {sub}: 사이클 CV={cvT*100:4.1f}% → 절대속도 하한 ±{cvT*C*3.6:5.1f}km/h | "
                  f"피치 CV={cvF*100:4.1f}% → ±{cvF*C*3.6:5.1f}km/h")


if __name__ == "__main__":
    main()

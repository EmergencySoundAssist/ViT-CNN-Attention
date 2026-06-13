"""
speed_baseline.py — 방식 A(물리 DSP) 속도 역산 베이스라인 (P2b 준비)

방식 B(DL head)가 넘어야 할 물리 베이스라인 곡선을 확정한다. 학습 불필요 — 추론/DSP만.

설계 (docs/03·04 합의):
  1. 정지(speed≈0) 사이렌 베이스 큐레이션:
     - acqMethod == '인위적' (제작=정지)  ← zero-state factor = 1.0 (라벨 모순 없음)
     - + f0 글라이드 검증: viterbi 트랙의 앞1/3 vs 뒤1/3 중앙값 차 < 임계 (잔여 운동 배제)
  2. synth_passby(v, d_min)로 알려진 v 주입 → estimate_passby로 역산
  3. v-오차 + 기권율(게이트가 None 반환)을 SNR × d_min 스윕으로 측정

핵심 비교 관점 (docs/04 §교차검증):
  - 방식 A는 모호하면 **기권**(None) → '오차'만이 아니라 '오차 vs 기권율 트레이드오프'로 평가
  - 합성-only 비교는 A에 home-advantage(생성 물리를 그대로 역산) → 진짜 칼날은 저SNR·d_min·실데이터 일치
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dataset as D
import doppler_speed as ds

RESULTS = Path(__file__).resolve().parent / "results"


def curate_stationary(max_n=120, min_dur=6.0, glide_max_hz=80.0, seed=42):
    """인위적 사이렌 중 f0 글라이드가 작은(정지) 클립 선별. (wav, sr, seg) 리스트."""
    src = D.split_sources(D.index_sources())
    cand = [s for s in src if s.label == "siren" and s.acq == "인위적" and s.dur >= min_dur]
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    pool, rejected = [], 0
    for s in cand:
        if len(pool) >= max_n:
            break
        try:
            sr, seg = ds.load_wav(s.wav, max_sec=12.0)
        except Exception:
            continue
        if len(seg) < sr * min_dur:
            continue
        t, f0 = ds.viterbi_f0(seg, sr)            # 문맥 추적 (배음 널뛰기 억제)
        n = len(f0)
        head, tail = np.median(f0[: n // 3]), np.median(f0[-n // 3:])
        if abs(head - tail) <= glide_max_hz:      # 글라이드 작음 = 정지
            pool.append((s.wav, sr, seg))
        else:
            rejected += 1
    print(f"정지 베이스: 인위적 사이렌 {len(cand)}개 중 {len(pool)}개 선별 "
          f"(글라이드>{glide_max_hz:.0f}Hz로 {rejected}개 배제)")
    return pool


def run_baseline(pool, speeds, snrs, d_mins, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for v in speeds:
        for d_min in d_mins:
            for tag, snr in snrs:
                errs, n_det, n_dir = [], 0, 0
                for wav, sr, seg in pool:
                    s2 = ds.synth_passby(seg, sr, v, d_min=d_min)
                    if snr is not None:
                        s2 = ds.add_noise(s2, snr, rng)
                    est = ds.estimate_passby(s2, sr)
                    if est is None:                    # 게이트 기권
                        continue
                    n_det += 1
                    errs.append(abs(est["v_kmh"] - v))
                    n_dir += est["direction"] == "approach→recede"   # 정답 방향
                rows.append({
                    "v": v, "d_min": d_min, "snr": tag,
                    "n_total": len(pool), "n_detected": n_det,
                    "abstain_rate": round(1 - n_det / len(pool), 3),
                    "med_err": round(float(np.median(errs)), 1) if errs else None,
                    "mean_err": round(float(np.mean(errs)), 1) if errs else None,
                    "dir_acc": round(n_dir / n_det, 3) if n_det else None,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="정지 베이스 클립 수")
    ap.add_argument("--smoke", action="store_true", help="작은 스윕")
    args = ap.parse_args()

    pool = curate_stationary(max_n=args.n)
    if not pool:
        print("정지 베이스 0개 — 큐레이션 기준 확인 필요")
        return 1

    if args.smoke:
        speeds, snrs, d_mins = [30, 60], [("clean", None), ("10dB", 10)], [8.0]
    else:
        speeds = [10, 20, 30, 40, 50, 60, 70, 80]
        snrs = [("clean", None), ("20dB", 20.0), ("10dB", 10.0), ("5dB", 5.0)]
        d_mins = [5.0, 15.0, 30.0]

    rows = run_baseline(pool, speeds, snrs, d_mins)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "speed_physics_baseline.json").write_text(
        json.dumps({"n_pool": len(pool), "rows": rows}, indent=1, ensure_ascii=False))

    # 요약 1: SNR × 속도 (중간 d_min 고정) — 오차·기권
    d_avail = sorted({r["d_min"] for r in rows})
    d_ref = d_avail[len(d_avail) // 2]               # 존재하는 중간 d_min
    print(f"\n{'='*70}\n방식 A 물리 역산 — 중앙값오차(km/h) | 기권율  [d_min={d_ref:.0f}m]\n{'='*70}")
    snr_tags = sorted({r["snr"] for r in rows}, key=lambda t: {"clean":0,"20dB":1,"10dB":2,"5dB":3}.get(t,9))
    print(f"{'v(km/h)':8s} " + " ".join(f"{t:>14s}" for t in snr_tags))
    for v in sorted({r["v"] for r in rows}):
        cells = []
        for t in snr_tags:
            m = next((r for r in rows if r["v"]==v and r["snr"]==t and r["d_min"]==d_ref), None)
            cells.append(f"{m['med_err']:>5}|{m['abstain_rate']*100:>4.0f}%" if m and m["med_err"] is not None else f"{'기권':>10s}")
        print(f"{v:<8d} " + " ".join(f"{c:>14s}" for c in cells))

    # 요약 2: d_min 영향 (clean, 속도 평균)
    print(f"\n차선거리 d_min 영향 (clean, 전 속도 중앙값오차 평균)")
    for d in sorted({r["d_min"] for r in rows}):
        es = [r["med_err"] for r in rows if r["d_min"]==d and r["snr"]=="clean" and r["med_err"] is not None]
        ab = [r["abstain_rate"] for r in rows if r["d_min"]==d and r["snr"]=="clean"]
        print(f"  d_min={d:>4.0f}m : 평균 중앙값오차 {np.mean(es):4.1f} km/h | 평균 기권율 {np.mean(ab)*100:3.0f}%")

    # 방향 정확도
    da = [r["dir_acc"] for r in rows if r["dir_acc"] is not None]
    print(f"\n방향(접근/이탈) 정확도: 평균 {np.mean(da)*100:.1f}% (n={len(da)} 셀)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

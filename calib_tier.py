"""calib_tier.py — 위험도 tier 경계(deadband/fast)·방향 정확도 캘리브레이션 하네스

실주행 라벨 녹음(CSV: wav,dir,v_kmh — dir∈{정지,접근,멀어짐}, v_kmh 선택)을 속도(+방향)
모델에 통과시켜 라벨별 v̂/방향 예측 분포를 뽑고, tier 경계를 데이터 기반으로 제안.
alert.py의 deadband=20/fast=40 placeholder를 실데이터로 교체하기 위한 도구.

  $ python calib_tier.py --csv drive_labels.csv                      # 실주행 데이터 오면
  $ python calib_tier.py --demo                                      # 데이터 없을 때: 합성
                                     # 통과(접근/멀어짐/정지)로 하네스+방향헤드 흐름 검증
⚠ --demo는 합성이라 in-domain — 실주행 캘리를 대체하지 않음(하네스 검증용).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

import dataset as ds
import infer

DIR_KO = ("정지", "접근", "멀어짐")
DIR_IDX = {k: i for i, k in enumerate(DIR_KO)}


@torch.no_grad()
def predict_file(y: np.ndarray, speed, device, stride_s: float = 1.0):
    """오디오 → (창별 v̂ 리스트, 창별 방향 idx 리스트|None)."""
    vs, dirs = [], []
    has_dir = getattr(speed, "dirhead", None) is not None
    for _, m in infer.windows(y, stride_s):
        x = torch.from_numpy(np.ascontiguousarray(infer._norm(m)))[None, None].to(device)
        out = speed(x)
        vs.append(float(out[0].cpu()))
        if has_dir:
            dirs.append(int(out[2][0].argmax().cpu()))
    return vs, (dirs if has_dir else None)


def summarize(rows):
    """rows: [(라벨dir, v_true|None, v̂중앙값, 방향예측|None)] → 리포트 + 경계 제안."""
    print(f"\n{'='*64}\n라벨별 v̂ 분포·방향 정확도 (파일 단위 중앙값/다수결)\n{'='*64}")
    by = {}
    for lab, vt, vm, dp in rows:
        by.setdefault(lab, []).append((vm, dp))
    for lab in DIR_KO:
        if lab not in by:
            continue
        vms = np.array([v for v, _ in by[lab]])
        dps = [d for _, d in by[lab] if d is not None]
        acc = (np.mean([d == DIR_IDX[lab] for d in dps]) * 100) if dps else None
        acc_s = f"  방향정확도 {acc:5.1f}%" if acc is not None else "  (방향헤드 없음)"
        print(f"  {lab:4s} n={len(vms):3d}  v̂ p5 {np.percentile(vms,5):5.1f} · p50 {np.percentile(vms,50):5.1f}"
              f" · p95 {np.percentile(vms,95):5.1f}{acc_s}")
    # 경계 제안: deadband = 정지 p95와 접근 p5 사이, fast = 접근 v̂ 상위 분기점(자료 충분할 때)
    if "정지" in by and "접근" in by:
        s95 = float(np.percentile([v for v, _ in by["정지"]], 95))
        a5 = float(np.percentile([v for v, _ in by["접근"]], 5))
        if s95 < a5:
            print(f"\ndeadband 제안: {(s95 + a5) / 2:.1f} km/h  (정지 p95 {s95:.1f} ↔ 접근 p5 {a5:.1f} 사이; 현재 20)")
        else:
            print(f"\n⚠ 정지 p95({s95:.1f}) ≥ 접근 p5({a5:.1f}) — v̂만으론 분리 불가."
                  f" 방향 헤드 판정을 주로 쓰고 deadband는 보조로.")
    print("(fast 경계는 접근 라벨에 v_kmh가 충분히 쌓이면 v_true~v̂ 회귀로 제안 가능)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="위험도 tier 경계 캘리")
    ap.add_argument("--csv", default=None, help="wav,dir[,v_kmh] 라벨 CSV")
    ap.add_argument("--demo", action="store_true", help="합성 통과로 하네스 흐름 검증")
    ap.add_argument("--speed-ckpt", default="models/speed_neural_dir.pt")
    args = ap.parse_args(argv)

    device = infer.pick_device()
    if not os.path.exists(args.speed_ckpt):
        print(f"체크포인트 없음: {args.speed_ckpt}"); return 1
    speed = infer.load_speed(args.speed_ckpt, device)
    print(f"속도 모델 {args.speed_ckpt} (방향헤드 {'있음' if getattr(speed,'dirhead',None) is not None else '없음'})")

    rows = []
    if args.csv:
        with open(args.csv, newline="") as f:
            for r in csv.DictReader(f):
                y = ds.load_wav(r["wav"])
                vs, dirs = predict_file(y, speed, device)
                if not vs:
                    continue
                vm = float(np.median(vs))
                dp = int(np.bincount(dirs).argmax()) if dirs else None
                rows.append((r["dir"].strip(), float(r["v_kmh"]) if r.get("v_kmh") else None, vm, dp))
    elif args.demo:
        import speed_neural as sn
        import speed_head as sh
        pool = sh.stationary_pool("test", 20, 7)
        print(f"demo: 합성 통과 (test 정지클립 {len(pool)}개) — in-domain 하네스 검증")
        for ph, lab in (("still", "정지"), ("approach", "접근"), ("recede", "멀어짐")):
            M, _, Y, _, _ = sn.gen(pool, 30, 7, ds.HOP, ds.N_FRAMES, snr=10.0, dir_head=True, phase=ph)
            with torch.no_grad():
                out = speed(torch.from_numpy(M)[:, None].to(device))
            has_dir = getattr(speed, "dirhead", None) is not None
            for k in range(len(M)):
                dp = int(out[2][k].argmax().cpu()) if has_dir else None
                rows.append((lab, float(Y[k]), float(out[0][k].cpu()), dp))
    else:
        ap.error("--csv 또는 --demo 필요")

    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""eval_hardneg.py — 닮은꼴(hard-negative) 오경보 평가 하네스 (S0 캘리 관문)

음악·사이렌FX·알람 등 "사이렌 닮은 비-사이렌" wav 폴더를 검출 파이프라인에 통과시켜
(a) 창 단위 마진 분포, (b) Gate 상태기계 시뮬 경보(ONSET) 발생, (c) τ_on 제안을 리포트.
alert.py의 τ_on=2.0·τ_crit=4.0 placeholder를 데이터로 검증/교체하기 위한 도구.

  $ python eval_hardneg.py --dir hardneg_wavs/     # 실데이터(음악/FX) 폴더
  $ python eval_hardneg.py --sanity                # 데이터 없을 때: test split horn/noise로
                                                   # 하네스 자체 검증 (FA≈0 기대)
⚠ --sanity는 학습과 같은 도메인이라 진짜 hard-negative 검증이 아님 — 하네스 동작 확인용.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

import alert
import dataset as ds
import infer


@torch.no_grad()
def margins_of(y: np.ndarray, det, device, stride_s: float, det_frames: int | None = None) -> list[float]:
    """오디오 → tick별 siren 마진 z[siren]-max(나머지). det_frames로 짧은 창(예비 채널) 평가."""
    out = []
    for _, m in infer.windows(y, stride_s):
        md = m if not det_frames else m[:, -det_frames:]
        x = torch.from_numpy(np.ascontiguousarray(infer._norm(md)))[None, None].to(device)
        z = det(x)[0].cpu().numpy()
        out.append(float(z[0] - max(z[1], z[2])))
    return out


def gate_onsets(margins: list[float], dt: float) -> int:
    """마진 시퀀스를 실제 Gate에 통과 → ONSET(오경보) 횟수."""
    g = alert.Gate(alert.CFG["siren"], dt)
    return sum(1 for m in margins if g.update(m)["onset"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hard-negative 오경보 평가")
    ap.add_argument("--dir", default=None, help="hard-negative wav 폴더 (음악/사이렌FX/알람)")
    ap.add_argument("--sanity", action="store_true", help="test split horn/noise로 하네스 검증")
    ap.add_argument("--ckpt", default="models/cnn_attn_full_s42.pt")
    ap.add_argument("--stride", type=float, default=0.5, help="tick 간격(초)")
    ap.add_argument("--window", type=float, default=None,
                    help="검출 창(초) — 예비 채널(2s) 오경보 평가용. 기본 5초 전체")
    ap.add_argument("--limit", type=int, default=200, help="sanity 파일 수 제한")
    args = ap.parse_args(argv)
    det_frames = (1 + int(args.window * ds.SR) // ds.HOP) if args.window else None

    device = infer.pick_device()
    det, name = infer.load_model(args.ckpt, None, device)
    det.eval()

    if args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "**", "*.wav"), recursive=True))
        src_desc = f"hard-negative 폴더 {args.dir}"
    elif args.sanity:
        src = ds.split_sources(ds.index_sources())
        files = [s.wav for s in src if s.split == "test" and s.label in ("horn", "noise")][: args.limit]
        src_desc = f"sanity(test split horn/noise, {len(files)}개) — 하네스 검증용, 진짜 hard-neg 아님"
    else:
        ap.error("--dir 또는 --sanity 필요")
    if not files:
        print("wav 없음"); return 1

    print(f"모델 {name} · {src_desc} · stride {args.stride}s"
          + (f" · 창 {args.window}s(예비 채널)" if args.window else " · 창 5s(확정 채널)"))
    all_m, total_sec, fa_files = [], 0.0, []
    for i, f in enumerate(files):
        try:
            y = ds.load_wav(f)
        except Exception:
            continue
        total_sec += len(y) / ds.SR
        mg = margins_of(y, det, device, args.stride, det_frames)
        if not mg:
            continue
        all_m.extend(mg)
        n_on = gate_onsets(mg, args.stride)
        if n_on:
            fa_files.append((f, max(mg), n_on))
        if (i + 1) % 50 == 0:
            print(f"  …{i+1}/{len(files)}", flush=True)

    m = np.array(all_m)
    hrs = total_sec / 3600
    tau = alert.CFG["siren"]["tau_on"]
    print(f"\n{'='*62}\n창 {len(m):,}개 · 오디오 {total_sec/60:.1f}분\n{'='*62}")
    print(f"마진 분위수: p50 {np.percentile(m,50):+.2f}  p95 {np.percentile(m,95):+.2f}"
          f"  p99 {np.percentile(m,99):+.2f}  p99.9 {np.percentile(m,99.9):+.2f}  max {m.max():+.2f}")
    print(f"창 단위 τ_on({tau}) 초과율: {(m >= tau).mean()*100:.3f}%")
    n_fa = sum(n for _, _, n in fa_files)
    print(f"Gate 시뮬 오경보(ONSET): {n_fa}회  →  FA/hour = {n_fa/hrs:.2f}" if hrs > 0 else "")
    if fa_files:
        print("오경보 유발 파일(마진 상위):")
        for f, mx, n in sorted(fa_files, key=lambda x: -x[1])[:10]:
            print(f"  max마진 {mx:+.2f} onset {n}회  {os.path.basename(f)}")
    # τ 제안: 비-siren p99.9 + 여유 0.5 (단 siren recall 쪽은 window_sweep 마진분포로 교차 확인 필요)
    sug = float(np.percentile(m, 99.9) + 0.5)
    print(f"\nτ_on 제안(비-siren p99.9+0.5): {max(sug, 0.5):.2f}  (현재 {tau})"
          f"\n⚠ siren쪽 여유 확인 필수: in-domain siren 마진 p5=+3.6 → 제안값이 그 아래여야 recall 유지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

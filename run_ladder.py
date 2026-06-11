"""
run_ladder.py — P1 검출 사다리 일괄 실행기 (중단·재개 안전)

results/ladder.json에 이미 기록된 (model, aug, seed) 조합은 건너뛰고 나머지를 순차 실행.
중간에 죽어도 다시 띄우면 이어서 돈다 (run 단위 재개 — run 내부 epoch 재개는 아님).

사용 (슬립 방지 + 세션 독립):
  nohup caffeinate -i python3 run_ladder.py --seed 42 > /tmp/airacle_ladder.log 2>&1 &
진행 확인:
  tail -f /tmp/airacle_ladder.log   또는   results/ladder.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 실행 계획 (docs/06 §6 P1 + ablation 2 증강 누적)
PLAN = [
    ("cnn", "none"), ("cnn_attn", "none"), ("vit", "none"),          # 무증강 사다리
    ("cnn_attn", "wave"), ("cnn_attn", "wave_spec"), ("cnn_attn", "full"),
    ("vit", "wave"), ("vit", "wave_spec"), ("vit", "full"),          # 증강 누적
]
LADDER = Path(__file__).resolve().parent / "results" / "ladder.json"


def done_set() -> set:
    try:
        hist = json.loads(LADDER.read_text())
        return {(r["model"], r.get("aug", "none"), r["seed"])
                for r in hist if not r.get("limit")}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for model, aug in PLAN:
        if (model, aug, args.seed) in done_set():       # 매 회 재조회 (run 끝날 때마다 갱신됨)
            print(f"skip {model}/{aug}/s{args.seed} — 완료", flush=True)
            continue
        log = f"/tmp/airacle_p1_{model}_{aug}_s{args.seed}.log"
        print(f"run  {model}/{aug}/s{args.seed} → {log}", flush=True)
        with open(log, "w") as f:
            r = subprocess.run(
                [sys.executable, "train.py", "--model", model, "--aug", aug,
                 "--seed", str(args.seed)],
                stdout=f, stderr=subprocess.STDOUT)
        print(f"  exit {r.returncode}", flush=True)
    print("사다리 전체 완료", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

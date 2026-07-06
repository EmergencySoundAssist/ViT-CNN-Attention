#!/usr/bin/env bash
# Airacle 검출 런타임 (Jetson) — 검출 경보 + 속도 위험도 + 차종(잠정)
#   ./run.sh           기본(경보 이벤트만 깔끔)
#   ./run.sh --debug   tick마다 상세(pred·margin·v̂)
#   ./run.sh --det-only  검출만(속도/차종 빼고)
# 주의: 시스템 python3(venv 금지). 마이크=ReSpeaker.
cd "$(dirname "$0")" || exit 1

# PulseAudio가 재꽂힌 ReSpeaker 캡처를 grab하면 직접-hw가 0-in이 됨 → 카드를 PulseAudio에서 해제.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
CARD=$(pactl list short cards 2>/dev/null | grep -i respeaker | head -1 | cut -f2)
if [ -n "$CARD" ]; then
    pactl set-card-profile "$CARD" off 2>/dev/null && echo "[run.sh] ReSpeaker를 PulseAudio에서 해제(캡처 hw 확보)"
fi

# 속도: 방향 헤드(_dir) 엔진이 빌드돼 있으면 우선(정지/멀어짐/접근 tier), 없으면 구엔진
SPEED="models/speed_neural.trt"
[ -f models/speed_neural_dir.trt ] && SPEED="models/speed_neural_dir.trt"
# 차종: 유튜브 채널 파인튜닝(_yt) 엔진 우선 — 실채널 held-out 1/5→3/5 (in-domain 89→86 트레이드)
SUBTYPE="models/subtype_cnn_attn_dom_s42.trt"
[ -f models/subtype_cnn_attn_yt_s42.trt ] && SUBTYPE="models/subtype_cnn_attn_yt_s42.trt"
ENGINES="--speed-engine $SPEED --subtype-engine $SUBTYPE"
# 예비검출(2s 창) 엔진 있으면 이중 창: PRE 예비경보 ≈2.7s + 확정 5.5s
[ -f models/cnn_attn_full_s42_87f.trt ] && ENGINES="$ENGINES --fast-engine models/cnn_attn_full_s42_87f.trt"
if [ "$1" = "--det-only" ]; then ENGINES=""; shift; fi

# stride 0.25: onset 지연 절반(게이트 1.0s→0.5s). 측정근거=정확도 손실 0, 연산 무시가능.
exec /usr/bin/python3 -u infer_trt.py --live --device ReSpeaker --stride 0.25 $ENGINES "$@"

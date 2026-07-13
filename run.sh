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
# 예비검출 엔진 있으면 이중 창: PRE 예비경보(1.5s창 ≈1.8s / 2s창 ≈2.7s) + 확정
FAST=""
[ -f models/cnn_attn_full_s42_87f.trt ] && FAST="models/cnn_attn_full_s42_87f.trt"
[ -f models/cnn_attn_full_s42_65f.trt ] && FAST="models/cnn_attn_full_s42_65f.trt"
[ -n "$FAST" ] && ENGINES="$ENGINES --fast-engine $FAST"
if [ "$1" = "--det-only" ]; then ENGINES=""; shift; fi

# stride 0.15: 게이트 확정 0.3s + PRE(1.5s창) 1.76s 실측. 연산: 검출 0.16ms×2+속도 12ms @6.7Hz — 여유.
# 자동 재시작: 안전제품에서 조용한 사망=무경보가 최악. 비정상 종료(크래시·오디오 워치독 code 3)만
# 재시작, 정상 종료(0, Ctrl-C 포함)는 그대로 끝. 장기적으론 systemd 서비스(Restart=always)로 승격 권장.
while :; do
    /usr/bin/python3 -u infer_trt.py --live --device ReSpeaker --stride 0.15 $ENGINES "$@"
    CODE=$?
    [ "$CODE" -eq 0 ] && exit 0
    echo "[run.sh] 런타임 종료 code=$CODE — 2s 후 재시작 (중단: Ctrl-C)" >&2
    sleep 2
done

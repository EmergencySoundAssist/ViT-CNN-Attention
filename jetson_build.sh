#!/usr/bin/env bash
# 젯슨에서 실행 — ONNX → TensorRT FP16 엔진 빌드 (이중창 검출 + 방향 속도).
# 선행: 맥에서 scp models/cnn_attn_full_s42_87f.onnx models/speed_neural_dir.onnx \
#              dcnm@192.168.55.1:~/Airacle/models/
# 빌드 후 ./run.sh 가 두 엔진을 자동 사용(파일 존재 감지).
set -e
cd "$(dirname "$0")"

for f in models/cnn_attn_full_s42_87f.onnx models/speed_neural_dir.onnx; do
    [ -f "$f" ] || { echo "없음: $f — 맥에서 scp 먼저"; exit 1; }
done

echo "[1/2] 예비검출(2s 창) 엔진…"
trtexec --onnx=models/cnn_attn_full_s42_87f.onnx \
        --saveEngine=models/cnn_attn_full_s42_87f.trt --fp16 > /dev/null
echo "[2/2] 속도+방향(_dir) 엔진…"
trtexec --onnx=models/speed_neural_dir.onnx \
        --saveEngine=models/speed_neural_dir.trt --fp16 > /dev/null
echo "완료 — ./run.sh 실행 (PRE 예비경보 ≈2.7s + 확정 5.5s, 위험도=정지/멀어짐/접근)"

# 08 · Jetson 실 테스트 런북 — 물리 vs 신경망 속도 (A/B)

> 목적: "나머지 다 동일, 추정기만 교체"로 **물리(estimate_passby) vs 순수 신경망(NeuralSpeed)**을
> 실제 Jetson Orin에서 **지연(latency) + 정확도 + A–B 일치율**로 비교. 브랜치 `jetson-speed-bench`.

## 0. 왜 실측이 필요한가 (Mac 사니티 결과)

Mac MPS 기준 — 물리 6 ms vs 신경망 **61 ms**(GRU가 병목, MPS가 RNN 최적화 안 함). Jetson은
CUDA/cuDNN/TensorRT라 GRU가 빨라질 수 있으나 **모름 → 실측**. (둘 다 관측시간 ~수초엔 미미하지만 측정 가치 있음.)

## 1. 전송 (Jetson으로)

```bash
# 로컬: 코드 + 캐시 + 정지 사이렌 wav 일부(정확도용; 지연만이면 생략 가능)
cd ~/PycharmProjects && tar -cf /tmp/airacle_jetson.tar \
  Airacle/*.py Airacle/cache Airacle/models Airacle/requirements.txt
# (정확도 측정하려면 인위적 사이렌 wav도 트리째 포함 — bench가 ds.load_wav로 읽음)
scp /tmp/airacle_jetson.tar <jetson>:/home/airacle/
```
Jetson: `tar -xf airacle_jetson.tar && cd Airacle`

## 2. 환경 (JetPack)

JetPack에 PyTorch(CUDA) 포함. 추가만:
```bash
pip install numpy scipy
python3 -c "import torch; print('cuda', torch.cuda.is_available())"   # True 확인
```

## 3. 지연 A/B (즉시 — 학습 가중치 없어도 지연은 유효)

```bash
python3 bench_speed.py --n 200          # 물리 vs 신경망 지연 중앙/p95
```
→ `device cuda` 로 떠야 함. GRU가 빠르면 신경망이 물리에 근접/우위, 느리면 4절(TensorRT)로.

## 4. 정확도·일치율 A/B (체크포인트 필요)

```bash
# 로컬에서 한 번 (또는 Jetson에서) 학습하며 저장:
python3 speed_neural.py --train 5000 --epochs 40 --save models/speed_neural.pt
# Jetson:
python3 bench_speed.py --ckpt models/speed_neural.pt --n 200
```
→ 물리 중앙오차·기권율 vs 신경망 중앙오차 + A–B 일치 중앙차. (합성 라벨 기준.)

## 5. TensorRT 최적화 (GRU 지연이 문제면)

```bash
python3 export_onnx.py --ckpt models/speed_neural.pt --out speed_neural.onnx
trtexec --onnx=speed_neural.onnx --fp16 --saveEngine=speed_neural.engine   # Jetson
```
TensorRT가 GRU를 융합·최적화 → 보통 큰 가속. **만약 TensorRT가 GRU를 잘 못 다루면** →
순환 GRU를 **TCN(시간 컨볼루션, 완전 병렬)** 으로 교체하는 fallback 검토(구조만 바꾸면 됨, 학습 방법 동일).

## 6. 진짜 마지막 관문 — 실도로 녹음

위는 전부 **합성** 기준. 4-mic 어레이로 **실제 사이렌 통과를 녹음** →
① 실음향에서 두 방식 A–B 일치율(sim-to-real 갭 측정) ② 차량 속도 메모로 희소 실라벨 ③ 데모.
이게 "합성에서 신경망이 물리 능가"가 **실도로에서도 유지되는지**의 최종 검증이다.

## 메모

- 지연 측정은 **멜 공유**(검출과 1회 계산) 가정 → 신경망은 forward만, 물리는 자체 STFT 포함(공정).
- `--fine`으로 학습했으면 bench도 `--fine` (해상도 일치). ckpt에 fine 플래그 저장됨 → 불일치 시 경고.
- 브랜치 격리: 배포·벤치 코드는 `jetson-speed-bench`, main은 깨끗하게 유지.

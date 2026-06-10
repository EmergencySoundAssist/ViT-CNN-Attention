# 00 · (아카이브) 최초 핸드오프 문서 — MIRACAR

> **이 문서는 프로젝트 시작 시점의 핸드오프 원본 보존본이다. 현재 설계가 아니다.**
> 이후 실측·분석으로 대체된 내용: RPi5 → **Jetson Orin**, 3초 윈도우 → **5초**(소방 wail 실측),
> 청크 후 8:1:1 split → **원본 단위 split**(누수 결함 수정, `dataset.py`),
> "ViT는 비교용" → **방식 A/B 동급 교차검증**, 배속 변환의 용도 재정의(도플러 시뮬 → 검출 불변성).
> 현재 설계는 [README](../README.md)와 docs/01~06 참조.

---

# MIRACAR 개발 핸드오프 문서

## 프로젝트 개요

**MIRACAR (MIRACAR Intelligent Real-time Audio Classification for Accessible Routing)**
청각장애 운전자를 위한 온디바이스 AI 긴급차량 감지 시스템.
마이크로 주변 소리를 실시간 분석하여 사이렌/경적을 감지하고, 시각+촉각 알림으로 전달.

- 대회: 2026 임베디드SW경진대회
- 팀명: Airacle
- 타겟 하드웨어: Raspberry Pi 5

---

## 시스템 아키텍처

```
마이크 입력 (22.05kHz)
    ↓
3초 버퍼 (슬라이딩 윈도우)
    ↓
멜 스펙트로그램 변환 (64 mel bands, n_fft=1024, hop=512)
    ↓
ONNX Runtime 추론 (CNN+Attention 모델)
    ↓
분류 결과: siren / horn / noise
    ↓
┌─ 시각 알림: LED / 디스플레이
├─ 촉각 알림: 시트 진동, 핸들 진동, 스마트워치 등 (부가기능)
└─ 방향 추정: 도플러 기반 긴급도 산출 (접근/이탈 판별)
```

### 핵심 기능
1. **사이렌 감지**: 긴급차량 사이렌 실시간 분류
2. **경적 감지**: 자동차 경적 감지
3. **도플러 기반 긴급도**: 주파수 변화율로 접근/이탈 판별, 긴급도 스코어 산출
4. **다중 알림**: 시각(LED/디스플레이) + 촉각(시트진동/핸들진동/스마트워치 등으로 추가 알림 발생 가능)

### 스마트워치 관련 참고
스마트워치는 부가기능. 핵심 알림이 아니라 "시트 진동, 스마트워치, 핸들 진동 등으로 추가 알림 발생 가능" 정도의 비중으로 처리.

---

## 데이터셋

### AI Hub "도시 소리 데이터"
- URL: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=585
- 총 73,864 샘플, 24 소분류, WAV 16bit 44.1kHz + JSON 라벨
- 다운로드 승인 필요 (회원가입 → 활용목적 작성 → 승인)

### MIRACAR에 필요한 클래스
| 클래스 | AI Hub 소분류 | 샘플 수 | 매핑 |
|--------|-------------|---------|------|
| siren | 사이렌 | 2,508 | 긴급차량 사이렌 |
| horn | 경적 (자동차 + 이륜차) | 4,001 + 5,594 | 자동차 경적 |
| noise | 나머지 22종 전부 | ~62,000 | 배경소음/비긴급음 |

### JSON 라벨 구조
```json
{
  "audio": { "fileName": "xxx.wav", "format": "WAV", "sampleRate": 44100 },
  "annotations": [{
    "categories": {
      "category_01": "교통소음",
      "category_02": "자동차",
      "category_03": "사이렌"    ← 이걸로 분류
    },
    "area": { "start": 0.0, "end": 5.2 },
    "decibel": 85.3
  }]
}
```

### 데이터 폴더 구조
```
data_root/
├── 원천데이터/           # WAV 파일
│   └── 교통소음/자동차/사이렌/*.wav
└── 라벨링데이터/          # JSON 파일 (1:1 매칭)
    └── 교통소음/자동차/사이렌/*.json
```

### 한국 사이렌 특성
- 법적으로 사이렌 "종류"를 규격화한 표준 없음
- 경찰/소방/구급이 주로 Yelp 패턴 사용, 톤/주파수로 구분
- 패턴 기반이 아닌 주파수 특성 기반 분류가 적합

---

## 데이터 증강 전략

### 문제: 사이렌 학습 데이터 ~2,000건 (8:1:1 split 후)

### 해결 1: 슬라이딩 윈도우
- 3초 윈도우, 1초 stride
- 5초 WAV → 3개 청크, 10초 → 8개 청크
- 예상 효과: 2,508건 → 약 7,000~15,000건

### 해결 2: 파형 레벨 증강 (5종)
| 증강 | 설명 | 목적 |
|------|------|------|
| 피치 시프트 | ±2 반음, 리샘플링 기반 | 다양한 사이렌 음높이 |
| 타임 스트레칭 | 0.85~1.15x, 피치 유지 | 사이렌 속도 변화 |
| **배속 변환** | 0.8~1.3x, 피치+속도 동시 변화 | **도플러 효과 시뮬레이션** |
| 볼륨 변화 | ±6dB | 거리에 따른 음량 차이 |
| 노이즈 합성 | SNR 5~20dB, noise 클래스와 합성 | 실제 도로 환경 |

**배속 변환 vs 피치 시프트 차이점:**
- 피치 시프트: 피치만 바꾸고 속도 유지 (리샘플링 후 길이 복원)
- 배속 변환: 피치+속도 동시 변환 → 사이렌 접근(높은 음+빠름) / 이탈(낮은 음+느림) 자연 모사

### 해결 3: 스펙트로그램 레벨 증강 (ViT용, 4종)
| 증강 | 설명 |
|------|------|
| SpecAugment | Time masking + Frequency masking |
| Mixup | 두 샘플 멜을 비율 혼합 + soft label (alpha=0.3) |
| CutMix | 랜덤 사각형 영역을 다른 샘플로 교체 |
| Random Erasing | 랜덤 영역을 0으로 마스킹 |

---

## 모델 아키텍처

### 최종 채택: CNN+Attention (~600K params)

**선택 근거:**
1. CNN의 귀납적 편향 (locality, translation invariance) → 소량 데이터에 강함
2. Temporal Attention → 사이렌 싸이클의 상승/하강/전환점에 가중치 학습
3. ~20ms 추론 → RPi5 실시간 요구사항 충족
4. ONNX 변환 완벽 지원

**구조:**
```
입력 (1, 64, T) 멜 스펙트로그램
  ↓
Conv2d Block × 3 (32→64→128 채널, 각 MaxPool2d)
  ↓
주파수 축 AdaptiveAvgPool → (128, T')
  ↓
Temporal Attention: Linear(128→64) → Tanh → Linear(64→1) → Softmax
  ↓
가중합 → (128,) context vector
  ↓
Classifier: Dropout → Linear(128→64) → ReLU → Linear(64→3)
  ↓
출력: [siren, horn, noise] 확률
```

### 비교 실험용: ViT (~800K params)

**목적:** "다른 모델도 시도해봤고, CNN+Attention이 우리 조건에 가장 적합했다"는 근거 확보

**구조:** 멜 스펙트로그램을 8×8 패치로 분할 → 패치 임베딩 → CLS 토큰 + Positional Embedding → Transformer Encoder (Pre-LN, 4 layers, 4 heads) → CLS 토큰으로 분류

**ViT가 이 프로젝트에서 약한 이유:**
- 귀납적 편향 없음 → 데이터가 수만 건은 돼야 CNN 수준 도달
- ~2,000건에서는 과적합 경향 (Mixup/CutMix로 완화 가능하나 근본적 한계)
- 추론 ~40ms로 CNN+Attention 대비 2배 느림

### 비교 대상에서 탈락한 모델들

| 모델 | 탈락 이유 |
|------|----------|
| CRNN (LSTM) | LSTM 순차 처리 → RPi5에서 ~80ms로 느림, ONNX 변환 까다로움 |
| CapsNet | AI Hub 베이스라인이지만 라우팅 연산이 RPi5에서 ~150ms, ONNX 지원 미흡 |
| 순수 RNN | 멜 스펙트로그램의 2D 구조를 활용 못함, CNN 대비 열등 |

---

## 학습 설정

```python
sample_rate = 22050       # 44.1kHz → 22.05kHz 다운샘플링
duration_sec = 3.0        # 3초 윈도우 (사이렌 1싸이클 ≈ 1~3초)
slide_stride_sec = 1.0    # 슬라이딩 윈도우 stride
n_mels = 64               # 멜 밴드 수
n_fft = 1024
hop_length = 512

batch_size = 32
epochs = 30
lr = 1e-3
weight_decay = 1e-4
optimizer = AdamW
scheduler = CosineAnnealingLR

split = 8:1:1 (Train:Val:Test, stratified)
loss = CrossEntropyLoss (inverse frequency 가중치로 클래스 불균형 대응)
```

---

## 추론 파이프라인 (RPi5)

```python
# 핵심 흐름
마이크 3초 녹음 (sounddevice, 22050Hz)
  ↓
멜 스펙트로그램 (NumPy only, librosa 없이 직접 구현 — import 속도 최적화)
  ↓
ONNX Runtime 추론 (CPUExecutionProvider)
  ↓
Softmax → argmax → confidence 체크 (threshold=0.7)
  ↓
결과에 따라 GPIO 제어 (LED, 모터 등)
```

### RPi5 추론 요구사항
- 추론 지연: < 50ms (CNN+Attention ~20ms로 충족)
- 전체 루프 (녹음 제외): < 100ms
- 메모리: ONNX 모델 ~2MB 이하
- 의존성 최소화: onnxruntime, numpy, sounddevice만

---

## 기존 산출물

### 코드 파일
- `train_pipeline.py` — 전체 학습 파이프라인 (데이터 로딩, 증강, CNN/CNN+Attn/ViT 모델, 학습루프, ONNX 내보내기)
- `inference_rpi5.py` — RPi5 추론 스크립트 (실시간 마이크 / WAV 파일 모드)

### 문서
- `03_2026ESWContest_MIRACAR_개발계획서_수정.docx` — 경진대회 개발계획서 (이미지 3장 포함, 8/10페이지)

### 다이어그램 (HTML)
- `다이어그램_시스템아키텍처.html` — 전체 시스템 아키텍처 흐름도
- `다이어그램_사이렌싸이클.html` — 사이렌 싸이클 분석 로직
- `다이어그램_비교분석표.html` — 유사 작품 비교 분석

---

## 개발 TODO

### Phase 1: 데이터 준비
- [ ] AI Hub 데이터 다운로드 승인 및 다운로드
- [ ] 데이터 로딩/파싱 검증 (train_pipeline.py의 parse_aihub_dataset)
- [ ] 슬라이딩 윈도우 적용 후 실제 샘플 수 확인
- [ ] 클래스별 분포 확인 및 noise 샘플링 비율 조정

### Phase 2: 모델 학습
- [ ] CNN 베이스라인 학습 → 성능 기록
- [ ] CNN+Attention 학습 → 성능 비교
- [ ] ViT 학습 (비교 실험) → 성능 비교
- [ ] 최적 모델 선정 및 하이퍼파라미터 튜닝
- [ ] ONNX 내보내기 및 추론 속도 측정

### Phase 3: RPi5 통합
- [ ] RPi5에 ONNX 모델 배포
- [ ] 마이크 입력 → 추론 → GPIO 알림 파이프라인 연결
- [ ] 실시간 추론 지연 측정 및 최적화
- [ ] 도플러 기반 긴급도 산출 로직 구현

### Phase 4: 알림 시스템
- [ ] LED/디스플레이 시각 알림 구현
- [ ] 진동 모터 촉각 알림 구현
- [ ] 알림 우선순위 로직 (siren > horn > noise)

---

## 핵심 기술 결정 요약

| 항목 | 결정 | 이유 |
|------|------|------|
| 모델 | CNN+Temporal Attention | 소량 데이터 강점 + 시간 패턴 + 엣지 추론 속도 |
| 추론 프레임워크 | ONNX Runtime | RPi5 CPU 최적화, 크로스플랫폼 |
| 오디오 입력 | 22.05kHz, 3초 윈도우 | 사이렌 1싸이클 커버, 메모리 효율 |
| 특징 추출 | 멜 스펙트로그램 64밴드 | 경량 + 충분한 주파수 해상도 |
| 데이터 증강 | 배속변환 + 노이즈합성 + Mixup | 소량 데이터 보완, 도플러 시뮬레이션 |
| 분류 체계 | 3클래스 (siren/horn/noise) | MIRACAR 목적에 필요충분 |

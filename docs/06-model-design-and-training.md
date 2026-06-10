# 06 · 모델 상세 설계 · 증강 근거 · 베이스라인 계획

> 개발 착수 전 확정 설계. 모든 증강 기법은 출처 논문과 "왜 우리 문제에 맞는지"를 함께 적는다.
> 실측 근거는 [docs/02](02-acoustic-analysis.md)·[docs/03](03-doppler-speed.md), 방식 A/B 구분은 [docs/04](04-architecture-and-comparison.md).

---

## 0. 공통 입력 사양

| 항목 | 값 | 근거 |
|------|-----|------|
| 샘플레이트 | 22,050 Hz | 사이렌 에너지 ≤2.5 kHz ([docs/02](02-acoustic-analysis.md)) — 나이퀴스트 여유 충분 |
| 윈도우 | **5.0 s** (slide 1.0 s) | 소방차 wail 1주기 ~4.7 s 포함 (실측) — 3 s는 부족 |
| 멜 스펙트로그램 | 64 mel · n_fft 1024 · hop 512 | 시간 프레임 T≈216, 경량·충분 해상도 |
| 정규화 | 클립별 (x−μ)/σ | 녹음 레벨 편차 제거 |
| 입력 텐서 | (1, 64, 216) | |

**분할 원칙 (누수 방지 — 중요):** 슬라이딩 청크 생성 **후**에 random split 하면 같은 원본 파일의 청크가 train/test 양쪽에 들어가 **누수**된다(기존 핸드오프 파이프라인의 결함). → **원본 파일 단위로 분할**한 뒤 청크 전개. AI Hub의 `2.Validation`은 **test 전용**으로 고정, `1.Training`을 파일 단위 90/10으로 train/val 분할(차종·클래스 stratify).

**클래스 균형:** siren 2,239 / horn 8,667 / noise 17,737. 전부 사용하되 ① siren은 슬라이딩으로 자연 증폭(~7–15×) ② WeightedRandomSampler로 배치 균형 ③ noise는 소분류(주행음/비행기/헬기/기차/지하철)별 stratify 유지 — 헬리콥터(주기성 hard negative)가 빠지지 않게.

---

## 1. 방식 A 검출 모델 — CNN + Temporal Attention (~0.6M)

### 1.1 구조 (layer-by-layer)

```
입력 (B, 1, 64, 216)
├─ Conv2d 3×3, 32  → BN → ReLU → MaxPool2   # (32, 32, 108)
├─ Conv2d 3×3, 64  → BN → ReLU → MaxPool2   # (64, 16, 54)
├─ Conv2d 3×3, 128 → BN → ReLU → MaxPool2   # (128, 8, 27)
├─ AdaptiveAvgPool(freq) → (128, 27) → permute (27, 128)
├─ Temporal Attention: Linear(128→64) → Tanh → Linear(64→1) → Softmax(T')
│    └ attention 가중합 → context (128,)
└─ Dropout 0.3 → Linear(128→64) → ReLU → Linear(64→3)
```

### 1.2 설계 근거 (논문)

| 설계 요소 | 출처 | 왜 우리 문제에 맞나 |
|----------|------|--------------------|
| 멜-CNN 백본 (3×3 conv 스택) | Salamon & Bello 2017, *Deep CNN and Data Augmentation for Environmental Sound Classification* (IEEE SPL) [arXiv:1608.04363] | 환경음 분류의 표준 백본. **UrbanSound8K에 siren 클래스 포함** — 가장 가까운 선행 과제. conv의 국소성·이동불변성이 소량 데이터(siren 2,239)에서 과적합 억제 |
| Temporal Attention (additive) | Bahdanau et al. 2015, *Neural Machine Translation by Jointly Learning to Align and Translate* (ICLR) [arXiv:1409.0473] — additive attention 원형 | Linear→Tanh→Linear 스코어링이 정확히 Bahdanau attention. 시간축 27 스텝 중 "사이클의 어느 구간이 증거인지" 가중 학습 |
| attention 풀링을 오디오 태깅에 적용 | Kong et al. 2018, *Audio Set classification with attention model* (ICASSP) | GAP(전구간 평균) 대비, 증거가 시간상 희소한 오디오(사이렌이 윈도우 일부에만 등장)에서 attention 풀링이 우월함을 보임 — 슬라이딩 청크 경계에서 사이렌이 잘릴 때 강건 |
| BN + MaxPool 경량 설계 | Piczak 2015, *ESC with CNN* (MLSP) | 환경음 CNN의 검증된 최소 구성. Orin에서 추론 ~수 ms |

**해석가능성 보너스**: attention 가중치 27개를 시각화하면 "모델이 윈도우의 어느 1초를 보고 siren이라 했는지" 데모 가능.

### 1.3 내부 ablation 베이스라인

attention의 기여를 입증하기 위해 **동일 백본 + GAP 풀링**(plain CNN, ~0.55M)을 함께 학습 — "attention이 실제로 뭘 더했나"의 직접 비교쌍.

---

## 2. 방식 B 검출 모델 — ViT (~0.8M)

### 2.1 구조

```
입력 (B, 1, 64, 216)
├─ Patch Embed: Conv2d k=8, s=8, 1→128     # 8×27 = 216 패치
├─ + CLS 토큰, + learnable pos embed (217)
├─ Transformer Encoder ×4 (Pre-LN, 4 heads, dim 128, MLP×4, dropout 0.1)
├─ LayerNorm → CLS 토큰
└─ Linear(128→64) → GELU → Dropout 0.2 → Linear(64→3)
```

### 2.2 설계 근거 (논문)

| 설계 요소 | 출처 | 왜 |
|----------|------|-----|
| 스펙트로그램에 ViT 적용 | Gong et al. 2021, *AST: Audio Spectrogram Transformer* (Interspeech) [arXiv:2104.01778] | 멜을 패치로 잘라 Transformer에 넣는 직계 선행. ESC-50/AudioSet SOTA. **단 AST는 ImageNet/AudioSet 사전학습에 크게 의존** → 우리는 from-scratch 경량이므로 증강이 생명줄 (아래 2.3) |
| ViT 원형 | Dosovitskiy et al. 2021, *An Image is Worth 16×16 Words* (ICLR) [arXiv:2010.11929] | 패치 임베딩 + CLS + pos embed 구성 그대로. 논문 스스로 "중간 규모 데이터에선 귀납적 편향 부재로 CNN에 밀림"을 명시 — 우리 비교 실험의 이론적 프레임 |
| Pre-LN 배치 | Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture* (ICML) [arXiv:2002.04745] | Post-LN 대비 warmup 민감도↓, 소규모 학습에서 안정 |
| 소량 데이터 ViT 학습 레시피 | Touvron et al. 2021, *DeiT: Training data-efficient image transformers* (ICML) [arXiv:2012.12877] | "사전학습 없이 ViT를 굴리려면 Mixup/CutMix/RandErasing 강증강 + AdamW + cosine"이라는 레시피의 출처. 우리 ViT 증강 정책이 이 논문을 따름 |

### 2.3 ViT 전용 학습 안정화

- 강증강 필수 (아래 §4 표의 7–10번) — DeiT 근거
- warmup 5 epoch → cosine decay
- (선택) stochastic depth 0.1, label smoothing 0.1
- (스트레치 골) DeiT-style distillation: CNN+Attn을 teacher로 — 두 방식의 시너지 데모

---

## 3. 방식 B 속도 head — synth_passby 학습

### 3.1 구조·학습

```
ViT backbone (검출과 공유, 동결) → CLS feature (128)
└─ 속도 head: Linear(128→64) → GELU → ┬ tier logits (3): 정지/접근-느림/접근-빠름
                                       └ direction logit (1): 접근/이탈
손실 = CE(tier) + λ·BCE(direction),  λ=0.5
```

**2-stage 학습**: ① 진짜 데이터로 검출 학습 → ② backbone **동결** 후 head만 synth_passby 합성셋으로 학습. 이유: 합성 도메인의 시프트가 검출 정확도를 오염시키지 않도록 격리. (head만으론 부족하면 마지막 encoder 블록만 미세조정)

### 3.2 합성 라벨 생성 (왜 이 방식인가 — [docs/03](03-doppler-speed.md) 요약)

- 균일 배속을 라벨로 쓰면 source 피치와 혼동 → **라벨 모순** → 학습 불가 (실측 ±50 km/h 바닥)
- `synth_passby`(retarded-time 통과 모델)는 비율 `(c+v)/(c−v)`가 **source-무관** → 라벨 일관 → 학습 가능
- 파라미터 랜덤화: v ∈ {0, 10~80 km/h}, d_min 5~30 m, SNR 5~20 dB(noise 클래스 합성), **윈도우 오프셋 랜덤**(통과 전·중·후 부분 글라이드 포함 — 실시간 5 s 윈도우가 전체 전이를 못 볼 수 있으므로)

### 3.3 관련 선행 연구 (속도 추정 자체)

| 출처 | 내용 | 우리와의 관계 |
|------|------|--------------|
| Cevher et al. 2009, *Vehicle Speed Estimation Using Acoustic Wave Patterns* (IEEE TSP) | 단일 마이크 통과 차량의 도플러+진폭 패턴으로 속도 추정 | 방식 A(DSP)의 물리 모델 선행. 우리는 사이렌 톤 구조(배음·사이클)를 추가 활용 |
| Marchegiani & Newman 2022, *Listening for Sirens* (IEEE T-ITS) [arXiv:1810.04989] | 도시 소음 속 사이렌 분리·분류 + 방향 | 사이렌 특화 검출·방향의 대표 선행. gammatonegram+CNN — 우리 비교군 설계에 참고 |
| Ko et al. 2015, *Audio Augmentation for Speech Recognition* (Interspeech) | 속도 섭동(리샘플) 증강의 원조 | "배속 변환" 기법 자체의 출처 — 단 우리는 **검출 불변성용**으로만, 라벨용으론 폐기 |

---

## 4. 증강 계획 — 기법 × 논문 × 왜

### 4.1 파형 레벨 (공통, 검출용)

| # | 기법 | 파라미터 | 출처 | 왜 우리 데이터에 필요한가 |
|---|------|---------|------|--------------------------|
| 1 | 슬라이딩 윈도우 | 5 s / stride 1 s | SED 표준 관행 (DCASE 계열) | siren 2,239 → 청크 ~15–20k. 시간 위치 불변성 |
| 2 | 피치 시프트 | ±2 semitone | Salamon & Bello 2017 | 실측 개체 산포 **피치 CV 16%** 커버 — 같은 차종도 사이렌 음높이가 다름 |
| 3 | 타임 스트레치 | 0.85–1.15× (피치 유지) | Salamon & Bello 2017 | 실측 사이클 산포(CV 4–7.5%, 소방 이봉) 커버 |
| 4 | 배속 변환 | 0.9–1.1× (피치+속도 동시) | Ko et al. 2015 (speed perturbation) | **도플러 불변성**: 접근/이탈로 스펙트럼 전체가 떠도 siren 인식. 0.9–1.1 = 물리적 ±120 km/h 상당 — 그 이상은 비물리라 축소 (기존 0.8–1.3에서 수정) |
| 5 | 게인 | ±6 dB | Salamon & Bello 2017 (DRC 변형) | 거리 1–80 m 산포, 마이크 감도 |
| 6 | 배경잡음 합성 | SNR 5–20 dB, noise 클래스에서 | Salamon & Bello 2017 | 실도로 SNR. **우리 noise 17,737개를 잡음원으로 재활용** — 외부 잡음 데이터 불필요 |

### 4.2 스펙트로그램 레벨 (ViT 위주 — DeiT 레시피)

| # | 기법 | 파라미터 | 출처 | 왜 |
|---|------|---------|------|-----|
| 7 | SpecAugment | time mask ≤20f, freq mask ≤8mel ×2 | Park et al. 2019 (Interspeech) [arXiv:1904.08779] | 시간/주파수 부분 가림 강건 — 경적·주행음이 사이렌을 부분 마스킹하는 상황 모사 |
| 8 | Mixup | α=0.3, soft label | Zhang et al. 2018 (ICLR) [arXiv:1710.09412] | ViT 과적합 억제 1순위 (DeiT 근거). 결정경계 평활 |
| 9 | CutMix | 영역 1/4–1/2 | Yun et al. 2019 (ICCV) [arXiv:1905.04899] | 시간-주파수 블록 교체 = **동시 발생음**(사이렌+경적) 시뮬레이션 |
| 10 | Random Erasing | p=0.4, 영역 1/8–1/3 | Zhong et al. 2020 (AAAI) [arXiv:1708.04896] | 순간 폐색(차체 가림, 마이크 클리핑) 강건 |

### 4.3 속도 라벨 생성용 (방식 B 전용)

| # | 기법 | 출처/근거 | 왜 |
|---|------|----------|-----|
| 11 | **synth_passby** | retarded-time 물리 모델 (자체 구현, [docs/03](03-doppler-speed.md)에서 검증 — 사이클 오차 1.1%) | **속도 라벨을 만들 유일하게 올바른 방법** (균일 배속은 라벨 모순) |
| 12 | (예정) 잔향 IR 합성 | Ko et al. 2017, *Reverberant speech augmentation* (ICASSP) | sim-to-real 1순위 갭(터널·빌딩협곡 잔향) 완화 |

### 4.4 적용 매트릭스

| | CNN+Attn (A) | ViT (B 검출) | 속도 head (B) |
|---|---|---|---|
| 1–6 파형 | ✅ | ✅ | 6만 (잡음) |
| 7–10 스펙트로그램 | 7만 (가볍게) | ✅ 전부 | ❌ (글라이드 패턴 보존) |
| 11–12 합성 | ❌ | ❌ | ✅ |

속도 head에 Mixup/마스킹을 안 쓰는 이유: 라벨(속도)이 **글라이드 모양 자체**에 있어서, 섞거나 가리면 라벨-입력 대응이 깨짐.

---

## 5. 베이스라인 · 평가 설계

### 5.1 비교 사다리 (내부, 동일 데이터·동일 증강)

| 레벨 | 모델 | 역할 |
|------|------|------|
| B0 | 다수결 (전부 noise) | 바닥 — 불균형 데이터에서 정확도 착시 점검 |
| B1 | Plain CNN (GAP 풀링, ~0.55M) | Salamon & Bello 2017 스타일 표준 베이스라인 |
| B2 | **CNN + Temporal Attention** | B1과의 차이 = attention의 기여 |
| B3 | **ViT** | B2와의 차이 = 전역 attention vs conv 편향 |
| (B4) | AST-small 전이 (선택) | 사전학습 상한선 참조 — "from-scratch가 어디까지 따라가나" |

### 5.2 외부 참조점 (직접 비교 불가 — 좌표로만)

| 출처 | 수치 | 주의 |
|------|------|------|
| AI Hub 도시소리 공식 베이스라인 (CapsNet 계열) | 데이터 설명서의 유효성 지표 — **TODO: 페이지에서 수치 확인** | 같은 데이터지만 24-클래스 과제라 3-클래스인 우리와 직접 비교 불가 |
| Salamon & Bello 2017 (UrbanSound8K, 10-class incl. siren) | 증강 CNN ~79% acc | 다른 데이터셋. siren 클래스 존재가 참고 포인트 |
| Tran & Tsai 2020, *Acoustic-Based EV Detection* (IEEE Access) | 자체 데이터 ~98% | 긴급차량 특화 선행 — 과제 정의가 가장 유사 |
| Asif et al. 2022, *LSSiren* (Scientific Data) | 사이렌/도로소음 공개셋 | (선택) **교차 데이터셋 일반화 테스트**용 외부 검증셋 후보 |

### 5.3 지표

- **주 지표: macro-F1** (클래스 불균형 — accuracy는 보조)
- **안전 지표: siren recall** — 운용점은 "siren recall ≥ 95%"로 threshold 고정 후, 그때의 **오경보율(FA/hour)** 을 noise 연속 스트림에서 측정 (놓친 사이렌 = 안전사고, 잦은 오경보 = 알림 무시 유발 — 양쪽 다 명시적 관리)
- 혼동행렬 (특히 **헬리콥터→siren** 오인 점검)
- 속도(B): tier 정확도·혼동행렬, 방향 정확도, A–B 일치율 ([docs/04](04-architecture-and-comparison.md) §평가 프로토콜)
- 효율: 파라미터 수, Orin 지연 (TensorRT FP16, batch=1, 1000회 중앙값)
- **3 seed 평균±표준편차** 보고 (단일 run 비교 금지)

### 5.4 학습 설정 (공통 고정)

```
optimizer  AdamW (lr 1e-3, wd 1e-4)
schedule   cosine, warmup 5ep (ViT) / 0ep (CNN)
batch      64
epochs     50, early stop (val macro-F1, patience 10)
loss       CE + inverse-frequency class weight
seed       {42, 43, 44}
```

---

## 6. 실행 계획 (단계 · 산출물)

| 단계 | 내용 | 산출물 | 완료 기준 |
|------|------|--------|----------|
| P0 | 데이터 파이프라인: 파일단위 split → 청크 인덱스 → 멜 캐시 | `dataset.py` | split 누수 0 검증 (원본파일 교집합 = ∅) |
| P1 | 검출 사다리: B1→B2→B3 학습 + 증강 ablation | `train.py`, 결과표 | 3-seed macro-F1, siren recall@운용점 |
| P2a | 방식 A 마무리: Viterbi phase-align + tier 래퍼 | `doppler_speed.py` 확장 | 합성 pass-by tier 정확도 |
| P2b | 방식 B 속도: synth_passby 셋 생성 + head 학습 | `speed_head.py` | 동일 셋 tier 정확도, A–B 일치율 |
| P3 | Orin 배포: ONNX → TensorRT FP16, 지연 측정 | `.engine`, 벤치 표 | end-to-end < 100 ms |
| P4 | 실시간 통합: 슬라이딩 추론 + 중앙값 + 알림 | `realtime.py` | 데모 시나리오 통과 |

### Ablation 목록 (보고서용 근거 생산)

1. attention 유무 (B1 vs B2)
2. 증강 누적: 없음 → +파형(1–6) → +Spec(7) → +Mixup/CutMix(8–10) — 모델별
3. 윈도우 3 s vs 5 s (소방 wail 실측 근거의 실증)
4. 속도: 균일배속 라벨 vs synth_passby 라벨 (라벨 모순 실증 — ±50 km/h 바닥 재현)
5. (선택) AST 전이 vs from-scratch

---

## 7. 리스크 & 선제 대응

| 리스크 | 신호 | 대응 |
|--------|------|------|
| ViT from-scratch 부진 | B3 < B1 | 증강 강화 확인 → patch 4×8(시간 해상도↑) → distillation(B2 teacher) |
| 헬리콥터 오경보 | 혼동행렬 heli→siren | noise 내 heli 비중 상향, hard negative mining |
| 속도 head sim-to-real | 실데이터 A–B 일치율 급락 | 잔향 IR 추가(#12), 합성 다양화, A 우선 운용 유지 |
| split 후 siren 부족 | val 분산 큼 | 슬라이딩 stride 0.5 s로 축소 (파일단위 split이라 누수 없음) |

---

## 참고문헌

1. Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Processing Letters*. arXiv:1608.04363
2. Piczak, K. J. (2015). Environmental Sound Classification with Convolutional Neural Networks. *MLSP*.
3. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*. arXiv:1409.0473
4. Kong, Q., et al. (2018). Audio Set Classification with Attention Model: A Probabilistic Perspective. *ICASSP*.
5. Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *ICLR*. arXiv:2010.11929
6. Gong, Y., Chung, Y.-A., & Glass, J. (2021). AST: Audio Spectrogram Transformer. *Interspeech*. arXiv:2104.01778
7. Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention (DeiT). *ICML*. arXiv:2012.12877
8. Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. *ICML*. arXiv:2002.04745
9. Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Interspeech*. arXiv:1904.08779
10. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*. arXiv:1710.09412
11. Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*. arXiv:1905.04899
12. Zhong, Z., et al. (2020). Random Erasing Data Augmentation. *AAAI*. arXiv:1708.04896
13. Ko, T., et al. (2015). Audio Augmentation for Speech Recognition. *Interspeech*.
14. Ko, T., et al. (2017). A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition. *ICASSP*.
15. Cevher, V., Chellappa, R., & McClellan, J. H. (2009). Vehicle Speed Estimation Using Acoustic Wave Patterns. *IEEE Transactions on Signal Processing*.
16. Marchegiani, L., & Newman, P. (2022). Listening for Sirens: Locating and Classifying Acoustic Alarms in City Scenes. *IEEE T-ITS*. arXiv:1810.04989
17. Tran, V.-T., & Tsai, W.-H. (2020). Acoustic-Based Emergency Vehicle Detection Using Convolutional Neural Networks. *IEEE Access*.
18. Asif, M., et al. (2022). Large-Scale Audio Dataset for Emergency Vehicle Sirens and Road Noises. *Scientific Data*.

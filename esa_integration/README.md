# esa_integration — ① 소리 분류 어댑터 (emergency-sound-assist 기여분)

팀 통합 repo [`emergency-sound-assist`](https://github.com/EmergencySoundAssist/emergency-sound-assist)의
**① 소리 분류 모듈**을, 이 repo(ViT-CNN-Attention)의 학습된 검출기로 채운 결과물이다.
팀 repo는 내 작업공간이 아니라 직접 못 올리므로, 여기 브랜치에 드롭인 패키지로 담는다.

> 담당 경계: ① 분류만. ② 방향(천자민)·③ 접근(김도윤)·pipeline은 손대지 않는다.

---

## 무엇인가

팀 계약 `classifier.infer(AudioChunk) → ClassResult`를, Airacle의 현역 검출기
`CNNAttn`(`cnn_attn_full_s42`, 3-클래스 {siren, horn, noise})로 구현한 **어댑터**.

계약(16kHz·1초 stateless 청크)과 모델 요구(22.05kHz·5초 윈도우)의 간극은
어댑터가 내부에서 흡수한다:

- 리샘플 16kHz → 22.05kHz (scipy 폴리페이즈, 없으면 선형 폴백)
- **5초 링버퍼**로 1초 청크 누적 → 최근 5초 창 분류(모델은 5초 문맥 필요)
- softmax → `ClassResult(label, confidence, is_emergency)` 매핑 (noise→normal_traffic)

## 파일 구성

```
esa_integration/
  classifier/
    infer.py            ← 어댑터(팀 계약 진입점). 링버퍼 은닉
    preprocessing.py    ← numpy logmel (Airacle dataset.py 추론경로 충실 추출)
    detector_model.py   ← 벤더링한 CNNAttn 정의
    weights/cnn_attn_full_s42.pt   ← 검출기 가중치(439KB)
    __init__.py
  core/types.py         ← 팀 계약 MIRROR (단독 테스트용, canonical은 팀 repo)
```

## 팀 repo에 넣는 법 (write 권한 있는 사람)

`esa_integration/classifier/`를 팀 repo의 `classifier/`에 그대로 복사. 단:

1. 팀 `.gitignore`가 `*.pt`를 막으므로 가중치 예외 추가:
   ```
   !classifier/weights/
   !classifier/weights/*.pt
   ```
2. `core/types.py`는 팀 repo 것을 쓴다(여기 사본 말고).
3. `docs/classifier/design.md`가 UrbanSound8K+MobileNet 계획으로 남아 있어
   실제(Airacle CNNAttn 벤더링)와 어긋남 → 현행화 필요(팀 논의).

## 테스트 (이 폴더에서)

```bash
cd esa_integration
python3 - <<'PY'
import numpy as np, core.types as T, classifier
r = classifier.infer(T.AudioChunk(samples=np.zeros(16000, np.float32)))  # 무음 1초
print(r.label.value, round(r.confidence, 3), r.is_emergency)
PY
```

검증 결과(팀 계약 경로, AI-Hub 샘플):

| 입력 | label | confidence | is_emergency |
|------|-------|-----------:|:---:|
| 사이렌 5s | `siren` | 1.000 | True |
| 경적 | `horn` | 0.999 | True |
| 무음 5s | `normal_traffic` | 0.828 | False |

## caveat

- **워밍업**: 5초 창 모델이라 스트림 시작 후 ~2초는 링버퍼가 덜 차 신뢰도 낮음
  (1초 문맥에선 사이렌이 horn으로 튀다가 2~3초에 락). 소스 전환 시 `classifier.reset()`.
- **경보 판정 제외**: 여긴 '분류'만. Airacle의 마진 임계·안정화 게이트(실경보 로직)는
  런타임/pipeline 몫이라 이 계약엔 포함하지 않는다.
- **의존성**: torch + numpy (+ scipy 있으면 고품질 리샘플).

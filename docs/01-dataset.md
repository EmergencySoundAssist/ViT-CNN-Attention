# 01 · 데이터셋

## 출처: AI Hub "도시 소리 데이터"

- URL: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=585
- 다운로드 **승인 필요** (회원가입 → 활용목적 작성 → 승인)
- 본 프로젝트가 쓰는 부분: **교통소음** 카테고리 (`130.도시 소리 데이터`)

> **라이선스·용량 때문에 데이터는 이 저장소에 포함하지 않습니다.** 직접 받아 아래 경로에 배치하세요.

## 폴더 구조 (표준 AI Hub 레이아웃)

```
130.도시 소리 데이터/01.데이터/
├── 1.Training/
│   ├── 라벨링데이터/TL_1.교통소음/...   # JSON
│   └── 원천데이터/TS_1.교통소음/...      # WAV
└── 2.Validation/
    ├── 라벨링데이터/VL_1.교통소음/...
    └── 원천데이터/VS_1.교통소음/...
```

소분류 9종: `1.차량경적` `2.차량사이렌` `3.차량주행음` `4.이륜차경적` `5.이륜차주행음` `6.비행기` `7.헬리콥터` `8.기차` `9.지하철`

## Airacle 3-클래스 매핑

| 클래스 | AI Hub 소분류 | 개수 |
|--------|--------------|------|
| **siren** | 차량사이렌 | 2,239 |
| **horn** | 차량경적 + 이륜차경적 | 3,588 + 5,079 = 8,667 |
| **noise** | 주행음·비행기·헬리콥터·기차·지하철 | 17,737 |
| | **합계** | **28,643** |

- 클래스 불균형(siren 1 : noise 8)에 주의 → 검출 학습 시 **noise/horn은 균형 잡힌 일부만** 샘플링 (전부 넣으면 학습 악화).
- 참고: **헬리콥터**(3,660)는 자체 주기성이 있어 사이렌과 헷갈리는 좋은 hard negative.

## JSON 라벨 구조 (핵심 필드)

```json
{
  "audio": { "fileName": "1.자동차_70450.wav", "sampleRate": "44.1kHz" },
  "environment": { "distance": "15m", "direction": "위", "acqMethod": "자연적" },
  "annotations": [{
    "labelName": "1.자동차_70450_1.wav",      // ← 실제 WAV 파일명 (audio.fileName과 다름!)
    "area": { "start": 2, "end": 31.24 },     // 사이렌 구간(초)
    "categories": { "category_03": "차량사이렌" },  // ← 클래스 분류 기준
    "subCategory": "경찰차"                    // 차종 (구급차/경찰차/소방차)
  }]
}
```

### ⚠️ 세 가지 함정 (반드시 처리)

1. **WAV 파일명은 `audio.fileName`이 아니라 `annotations[0].labelName`** (뒤에 `_1`이 붙음).
   `audio.fileName`으로 매칭하면 **0건**이 됩니다.

2. **macOS 파일시스템(NFD) ↔ JSON 문자열(NFC) 한글 불일치.**
   파일시스템의 `자동차`(NFD)와 JSON의 `자동차`(NFC)는 바이트가 달라 dict 매칭이 실패합니다.
   → 모든 비교 전 `unicodedata.normalize("NFC", ...)` 필수.

3. **`area.start/end`는 원본 녹음 좌표 — labelName wav 내부 오프셋이 아님.**
   원천 wav는 **이미 area 구간만 잘라낸 클립**입니다 (원본:클립 = 1:1, end−start ≈ wav 길이 — 전수 확인).
   start/end로 wav를 다시 자르면 앞 ~2초(전형 start=2)를 버리게 됩니다. 클립 분석·청크 그리드는 wav 전체에.

## 메타데이터 분포 (전수 확인)

- `distance`: 1m~80m, 대부분 10m — **고정 거리 녹음**
- `direction`: 위(1194)/우(822)/좌(182)/아래(41) — 마이크 기준 **방위**(움직임 방향 아님)
- `acqMethod`: 자연적(1604)/인위적(635)
- `subCategory`: 구급차(1145)/경찰차(569)/소방차(525)
- duration 중앙값 16.7s, 사이렌 구간 중앙값 11.1s

### 이것이 설계에 주는 결정적 제약

녹음이 **고정 거리·고정 방위**이고 `acqMethod`가 제작/자연이라, **실제 통과(pass-by) 도플러 전이가 라벨돼 있지 않습니다.** 즉:
- **속도·방향의 정답(ground truth)이 데이터에 없음.**
- → 방향·속도를 지도학습하려면 **합성(`synth_passby`)**으로 라벨을 만들어야 함.
- → 그래서 무학습 물리 DSP가 일차 선택. ([docs/03](03-doppler-speed.md))

## 로더: `siren_data.py`

```python
from siren_data import index_clips
clips = index_clips(classes=("siren",))   # 또는 ("siren","horn","noise")
# 각 Clip: wav 경로, label, sub(차종), start/end(⚠ 원본 좌표 — 함정 #3), split(AI Hub 폴더 기준)
```

NFC 정규화와 `labelName` 매칭을 내부에서 처리. `classes=` 인자 한 줄로 3-클래스 확장.
**학습용 train/val/test 분할의 정본은 `dataset.py`** (원본 단위 split + 청크 + 멜 캐시) — siren_data의 split은 AI Hub 폴더 구분일 뿐.

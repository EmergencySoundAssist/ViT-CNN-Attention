# 각도 → 4방향 매핑 (`angle_to_direction`)

> DoA 1단계(MVP)의 핵심 순수 로직. ReSpeaker raw DOA(0~359°)를 **차량 기준** 전/후/좌/우로 변환한다.
> 구현: [`doa/estimator.py`](../../doa/estimator.py) · 출력 타입: [`core/types.py`](../../core/types.py) `Direction`
> 상위 설계: [design.md](./design.md) · 하드웨어: [hardware.md](../hardware.md)

---

## 1. 무슨 함수인가

```python
def angle_to_direction(raw_deg: float) -> Direction
```

- **입력**: `raw_deg` — ReSpeaker `Tuning.direction`이 주는 0~359° 정수(또는 테스트용 주입값). 음수·360 이상도 `% 360`으로 정규화하므로 안전.
- **출력**: `core.types.Direction` 중 하나 — `FRONT` / `REAR` / `LEFT` / `RIGHT`.
  (각도가 없을 때의 `UNKNOWN`은 상위 `estimate_direction`이 처리하며, 이 함수는 항상 4방향 중 하나를 반환한다.)
- **부수효과 없음 / 장치 불필요** → 노트북에서 단위 테스트 가능.
- 내부적으로 `_to_vehicle_angle()`로 **장착 보정**을 먼저 한 뒤 사분면을 가른다(2절).

---

## 2. 2단계 변환: raw DOA → 차량 기준 → 4방향

### (a) ReSpeaker raw DOA 규약 (보드 인쇄 기준)
보드에 인쇄된 `0/90/180/270` 마크가 펌웨어 기준점이다. **0°=보드 우측, 반시계(CCW)로 증가.**
```
            raw 90° (보드 위)
                 ↑
raw 180° ←     XMOS     → raw 0°  (보드 우측)
                 ↓
            raw 270° (보드 아래)
```

### (b) 장착 보정 — 케이블 = 후방
보드를 차량에 장착하면 raw 각도를 그대로 못 쓴다. **케이블이 가리키는 방향을 후방으로** 삼아 차량 기준으로 재정렬한다. 보정은 상수 2개에 모인다:

| 상수 | 의미 |
|---|---|
| `REAR_RAW_DEG` | 케이블이 가리키는 raw 각도. 이 방향을 후방(180°)으로 맞춤. |
| `MIRROR` | 좌/우 반전 보정. 보드 CCW ↔ 차량 기준 좌/우 차이를 흡수. |

```python
def _to_vehicle_angle(raw_deg):
    deg = raw_deg % 360.0
    if MIRROR:
        deg = (360.0 - deg) % 360.0
    cable = (360.0 - REAR_RAW_DEG) % 360.0 if MIRROR else REAR_RAW_DEG
    return (deg - cable + 180.0) % 360.0   # 케이블을 180°(후방)로 이동
```

> 현재 기본 보정값: **`REAR_RAW_DEG = 270` (케이블이 보드 아래쪽), `MIRROR = True`.**
> ⚠️ **실측으로 확정 필요** — 4절 보정 절차 참고. (장착 방향이 바뀌면 이 두 값만 다시 잡으면 됨.)

### (c) 차량 기준 각도 → 4방향
`_to_vehicle_angle` 출력은 **표준 나침반 순서**(전 0° / 우 90° / 후 180° / 좌 270°). 경계는 **반개구간 `[start, end)`**.

| 차량 기준 각도 | 반환 `Direction` |
|---|---|
| `315° ~ 360°` 및 `0° ~ 45°` | `FRONT` |
| `45° ~ 135°` | `RIGHT` |
| `135° ~ 225°` | `REAR` |
| `225° ~ 315°` | `LEFT` |

```
          전방 0° / 360°
              FRONT
        315° ┌──┴──┐ 45°
        LEFT │     │ RIGHT
        225° └──┬──┘ 135°
              REAR  ← 케이블
             180°
```

> 각 방향은 ±45° 폭(90°)을 차지. 경계값 귀속: `45° → RIGHT`, `135° → REAR`, `225° → LEFT`, `315° → FRONT`.
> **장착 quirk(회전·반전)는 전부 (b)에 모이고, (c) 사분면 경계는 절대 안 건드린다.**

---

## 3. 예시 (기본 보정값 `REAR_RAW_DEG=270, MIRROR=True` 기준)

기본 보정에서는 `차량각 = (90 − raw) mod 360`.

| raw DOA (보드 위치) | 차량 기준 각도 | 결과 |
|---|---|---|
| `90` (보드 위) | `0` | `FRONT` |
| `0` (보드 우) | `90` | `RIGHT` |
| `270` (보드 아래 = 케이블) | `180` | `REAR` |
| `180` (보드 좌) | `270` | `LEFT` |

`python -m doa.estimator` 로 raw → 차량각 → 방향을 한 번에 확인 가능.

---

## 4. 보정값 확정 절차 (실측 — 가장 중요)

머리로 추론하지 말고 **마이크에 직접 물어본다.** 보드를 차량 장착 방향대로 고정한 뒤:

```bash
cd emergency-sound-assist
python3 doa/respeaker_tuning.py DOAANGLE   # 소리 내면서 여러 번 실행
```

1. **후방(케이블 쪽)에서** 소리 → 나온 raw 값을 `REAR_RAW_DEG`에 기입.
2. **우측에서** 소리 → `angle_to_direction(raw)` 결과 확인:
   - `RIGHT` 나오면 → `MIRROR` 맞음 ✅
   - `LEFT` 나오면 → `MIRROR` 토글 🔁
3. 전·후·좌·우 4방향을 모두 쏴서 한 바퀴 검증.

> 확정 전까지 `estimator.py`의 두 상수에 `TODO(보정)` 주석을 남겨둔다.

---

## 5. 상위 흐름에서의 위치

```
estimate_direction(chunk, angle_deg=None)
  ├─ angle_deg 주입됐으면 그 값 사용 (테스트용)
  ├─ 아니면 ReSpeaker 폴링 (_read_respeaker_angle)
  ├─ 둘 다 None → DirectionResult(UNKNOWN, angle_deg=None)
  └─ 각도 있으면 → angle_to_direction(angle_deg) → DirectionResult(dir, angle_deg)
```

- `angle_deg`(및 `DirectionResult.angle_deg`)는 **보정 전 raw 값**을 그대로 보존한다(디버깅·2단계용). 변환은 `angle_to_direction` 내부에서만 일어난다.
- `chunk`(4채널 오디오)는 1단계에선 미사용이며 2단계 GCC-PHAT에서 활용 예정.

---

## 6. 리뷰 체크리스트

- [x] 좌/우 반전 → `MIRROR` 상수로 흡수 (실물 테스트에서 좌/우 뒤집힘 확인됨)
- [x] 케이블=후방 규약 → `REAR_RAW_DEG`로 파라미터화, 사분면 경계는 나침반 순서 고정
- [ ] **`REAR_RAW_DEG` / `MIRROR` 실측 확정** (4절) — 현재 기본값 `270 / True`는 추정치
- [ ] 마이크 어레이 차량 장착 방향(케이블이 정확히 후방인지) 물리적으로 고정했는가
- [ ] 단위 테스트: 4방위 raw(0/90/180/270) + 경계 직전 케이스 + `_to_vehicle_angle` 검증 포함

"""
② 방향 추정 모듈.

노트북 단계: ReSpeaker 실물이 없으므로 각도→4방향 변환 로직만 동작한다.
Jetson 단계: ReSpeaker `Tuning.direction`(0~359°)을 폴링해 `angle_deg`에 채운다.

- 입력: 4채널 AudioChunk (ReSpeaker 원시 오디오)  ※ 1단계에선 사용 안 함
- 출력: DirectionResult (전/후/좌/우/unknown)
"""

from __future__ import annotations

from typing import Optional

from core.types import AudioChunk, Direction, DirectionResult

# ---------------------------------------------------------------------------
# ReSpeaker 의존성 (있으면 쓰고, 없으면 조용히 비활성화)
# ---------------------------------------------------------------------------
# respeaker_tuning.find() 가 내부에서 pyusb 를 쓰므로, pyusb 가 없으면
# 아래 import 자체가 ImportError 로 떨어진다 → 노트북에서도 안전하게 비활성화.
try:
    from .respeaker_tuning import find as _respeaker_find  # type: ignore
    _RESPEAKER_LIB_AVAILABLE = True
except ImportError:
    _RESPEAKER_LIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# 보드 장착 보정 (실측으로 확정 — docs/doa/direction-mapping.md 4절 참고)
# ---------------------------------------------------------------------------
# ReSpeaker raw DOA 규약: 0°=보드 우측, 반시계(CCW) 증가 (보드 인쇄 0/90/180/270).
# 케이블이 가리키는 방향을 '후방'으로 삼아 차량 기준으로 재정렬한다.
#   REAR_RAW_DEG : 케이블이 가리키는 raw 각도. 후방(180°)에 맞춘다.
#   MIRROR       : 좌/우 반전 보정. 실측에서 좌우 반대로 나오면 토글.
# TODO(보정): 차량 장착 상태에서 4방향 실측 후 아래 두 값을 확정할 것.
REAR_RAW_DEG = 270.0
MIRROR = True


def _to_vehicle_angle(raw_deg: float) -> float:
    """raw DOA → 차량 기준 각도(전 0° / 우 90° / 후 180° / 좌 270°).

    케이블 방향(`REAR_RAW_DEG`)을 후방(180°)에 맞추고, 필요시 좌/우를 반전한다.
    장착에서 비롯되는 회전·반전은 모두 이 함수(+상수 2개)에 모이고,
    `angle_to_direction` 의 사분면 경계는 표준 나침반 순서로 고정된다.
    """
    deg = raw_deg % 360.0
    if MIRROR:
        deg = (360.0 - deg) % 360.0
    cable = (360.0 - REAR_RAW_DEG) % 360.0 if MIRROR else REAR_RAW_DEG
    return (deg - cable + 180.0) % 360.0


def angle_to_direction(raw_deg: float) -> Direction:
    """raw DOA(0~359°) → 4방향. 케이블=후방 기준. 경계는 [start, end) 반개구간.

    먼저 `_to_vehicle_angle` 로 차량 기준(전 0/우 90/후 180/좌 270)으로 바꾼 뒤
    ±45° 폭으로 사분면을 가른다.
    """
    deg = _to_vehicle_angle(raw_deg)
    if 45.0 <= deg < 135.0:
        return Direction.RIGHT
    if 135.0 <= deg < 225.0:
        return Direction.REAR
    if 225.0 <= deg < 315.0:
        return Direction.LEFT
    return Direction.FRONT  # 315~360 또는 0~45 (차량 기준 전방)


def _read_respeaker_angle() -> Optional[float]:
    """ReSpeaker XVF-3000 의 자체 DoA 값(0~359°)을 읽는다.

    아래 세 경우 모두 None 반환 (노트북 단계 정상 경로):
        · pyusb / respeaker_tuning import 실패
        · USB 장치 미감지 (마이크 안 꽂힘)
        · ※ 실물 검증은 Jetson 단계에서 처음 이뤄짐
    """
    if not _RESPEAKER_LIB_AVAILABLE:
        return None
    mic = _respeaker_find()          # 장치 못 찾으면 None 반환
    if mic is None:
        return None
    return float(mic.direction)


def estimate_direction(
    chunk: AudioChunk,
    angle_deg: Optional[float] = None,
) -> DirectionResult:
    """4채널 오디오 → 방향.

    우선순위:
        1) 호출 측에서 `angle_deg` 를 직접 주면 그 값을 사용 (테스트/주입용)
        2) 아니면 ReSpeaker 에서 폴링 시도
        3) 둘 다 실패하면 UNKNOWN

    `chunk` 자체는 1단계(MVP)에선 사용하지 않으며, 2단계 GCC-PHAT 에서 활용 예정.
    """
    if angle_deg is None:
        angle_deg = _read_respeaker_angle()
    if angle_deg is None:
        return DirectionResult(direction=Direction.UNKNOWN, angle_deg=None)
    return DirectionResult(direction=angle_to_direction(angle_deg), angle_deg=angle_deg)

if __name__ == "__main__":
    print(f"보정: REAR_RAW_DEG={REAR_RAW_DEG}, MIRROR={MIRROR}")
    print("raw → 차량각 → 방향")
    for a in [0, 90, 180, 270, 44, 45, 359]:
        print(f"  {a:>3} → {_to_vehicle_angle(a):>5.0f} → {angle_to_direction(a).value}")
    # ReSpeaker 미연결 시 UNKNOWN, 연결 시 실측 각도 출력
    print("주입 없음 →", estimate_direction(None))
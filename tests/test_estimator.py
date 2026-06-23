"""doa.estimator 단위 테스트.

두 그룹으로 나눈다:
  · 불변 계약 (보정값 REAR_RAW_DEG/MIRROR 가 바뀌어도 항상 성립)
  · 현재 기본 보정값 기준 (REAR_RAW_DEG=270, MIRROR=True; 실측 확정 시 갱신)

실물(ReSpeaker) 연결 여부와 무관하도록, 폴링 경로는 monkeypatch 로 격리한다.
실행: 레포 루트에서 `pytest tests/test_estimator.py`
"""

import pytest

from core.types import Direction
from doa import estimator
from doa.estimator import (
    REAR_RAW_DEG,
    angle_to_direction,
    estimate_direction,
    _to_vehicle_angle,
)


# ---------------------------------------------------------------------------
# 그룹 1: 불변 계약 — 보정값과 무관하게 항상 성립
# ---------------------------------------------------------------------------

def test_cable_direction_is_rear():
    """케이블 방향(REAR_RAW_DEG)은 정의상 항상 후방."""
    assert _to_vehicle_angle(REAR_RAW_DEG) == pytest.approx(180.0)
    assert angle_to_direction(REAR_RAW_DEG) is Direction.REAR


def test_front_is_opposite_of_cable():
    """전방은 케이블 반대편(+180°)."""
    assert angle_to_direction((REAR_RAW_DEG + 180.0) % 360.0) is Direction.FRONT


def test_normalization_wraps_360():
    """음수·360 이상 각도도 % 360 으로 동일 처리."""
    for raw in [0.0, 37.0, 123.5, 270.0, 359.0]:
        assert angle_to_direction(raw) is angle_to_direction(raw + 360.0)
        assert angle_to_direction(raw) is angle_to_direction(raw - 360.0)


def test_always_returns_a_cardinal_direction():
    """각도가 주어지면 절대 UNKNOWN 이 아니라 4방향 중 하나."""
    cardinal = {Direction.FRONT, Direction.REAR, Direction.LEFT, Direction.RIGHT}
    for raw in range(0, 360, 5):
        assert angle_to_direction(float(raw)) in cardinal


def test_left_right_are_mirror_opposites():
    """좌/우는 서로 반대편(±180°)에 위치."""
    # 우측으로 판정되는 각을 찾아 +180 하면 좌측이어야 함
    right_raw = next(
        r for r in range(360) if angle_to_direction(float(r)) is Direction.RIGHT
    )
    assert angle_to_direction(float((right_raw + 180) % 360)) is Direction.LEFT


# ---------------------------------------------------------------------------
# 그룹 2: 현재 기본 보정값 기준 (REAR_RAW_DEG=270, MIRROR=True)
#   ⚠️ 실측으로 보정값 확정 시 이 그룹을 갱신할 것.
# ---------------------------------------------------------------------------

requires_default_calibration = pytest.mark.skipif(
    not (REAR_RAW_DEG == 270.0 and estimator.MIRROR is True),
    reason="기본 보정값(270/True)이 아님 — 실측 보정 후 기대값 갱신 필요",
)


@requires_default_calibration
@pytest.mark.parametrize(
    "raw, vehicle, direction",
    [
        (90.0, 0.0, Direction.FRONT),    # 보드 위
        (0.0, 90.0, Direction.RIGHT),    # 보드 우
        (270.0, 180.0, Direction.REAR),  # 보드 아래 = 케이블
        (180.0, 270.0, Direction.LEFT),  # 보드 좌
    ],
)
def test_default_cardinals(raw, vehicle, direction):
    assert _to_vehicle_angle(raw) == pytest.approx(vehicle)
    assert angle_to_direction(raw) is direction


@requires_default_calibration
@pytest.mark.parametrize(
    "raw, direction",
    [
        (44.0, Direction.RIGHT),   # 차량각 46° → RIGHT
        (45.0, Direction.RIGHT),   # 차량각 45° → 경계(45 포함) → RIGHT
        (46.0, Direction.FRONT),   # 차량각 44° → FRONT
        (135.0, Direction.FRONT),  # 차량각 315° → 경계(315 제외) → FRONT
        (136.0, Direction.LEFT),   # 차량각 314° → LEFT
    ],
)
def test_default_boundaries(raw, direction):
    assert angle_to_direction(raw) is direction


# ---------------------------------------------------------------------------
# 그룹 3: estimate_direction 흐름 (폴링 경로 격리)
# ---------------------------------------------------------------------------

def test_injected_angle_takes_priority(monkeypatch):
    """angle_deg 주입 시 폴링하지 않고 그 값을 쓴다."""
    monkeypatch.setattr(estimator, "_read_respeaker_angle", lambda: 999.0)
    res = estimate_direction(None, angle_deg=REAR_RAW_DEG)
    assert res.direction is Direction.REAR
    assert res.angle_deg == pytest.approx(REAR_RAW_DEG)  # raw 값 그대로 보존


def test_unknown_when_no_angle(monkeypatch):
    """주입도 없고 폴링도 실패하면 UNKNOWN."""
    monkeypatch.setattr(estimator, "_read_respeaker_angle", lambda: None)
    res = estimate_direction(None)
    assert res.direction is Direction.UNKNOWN
    assert res.angle_deg is None


def test_polling_used_when_not_injected(monkeypatch):
    """주입이 없으면 폴링 값을 사용."""
    monkeypatch.setattr(estimator, "_read_respeaker_angle", lambda: REAR_RAW_DEG)
    res = estimate_direction(None)
    assert res.direction is Direction.REAR
    assert res.angle_deg == pytest.approx(REAR_RAW_DEG)

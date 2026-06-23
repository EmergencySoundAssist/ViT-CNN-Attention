"""
공통 데이터 인터페이스 (팀 전체가 공유하는 '약속').

세 기능(분류 / 방향 / 접근)이 각각 독립적으로 개발되더라도,
여기 정의된 형식만 지키면 pipeline 단계에서 문제없이 합쳐진다.

- 오디오 입력 형식: AudioChunk
- 각 모듈의 출력 형식: ClassResult / DirectionResult / ApproachResult
- 최종 통합 결과: FusedResult  (예: "사이렌, 후방, 접근 중")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 공통 오디오 설정 (모든 모듈이 동일하게 사용)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # Hz. 음성/환경음 분류에서 일반적으로 충분한 값.
CHUNK_SECONDS = 1.0           # 한 번에 분석하는 오디오 길이(초).
CHANNELS = 4                  # ReSpeaker XVF-3000 = 4채널. 분류는 보통 1채널로 다운믹스.


@dataclass
class AudioChunk:
    """마이크에서 들어온 한 조각의 오디오.

    samples: shape (n_samples,) 단일 채널 또는 (n_samples, channels) 다채널.
             분류 모듈은 단일 채널만 쓰면 되고, 방향(doa) 모듈이 다채널을 쓴다.
    """
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE


# ---------------------------------------------------------------------------
# ① 소리 분류 모듈 출력
# ---------------------------------------------------------------------------
class SoundClass(str, Enum):
    SIREN = "siren"
    HORN = "horn"
    NORMAL_TRAFFIC = "normal_traffic"
    # 이후 데이터가 쌓이면 ambulance / fire_truck / police 로 세분화 확장.


@dataclass
class ClassResult:
    """분류 모듈(★ 내 담당)의 출력."""
    label: SoundClass
    confidence: float                       # 0.0 ~ 1.0
    is_emergency: bool                       # siren/horn 이면 True

    @staticmethod
    def from_label(label: SoundClass, confidence: float) -> "ClassResult":
        emergency = label in (SoundClass.SIREN, SoundClass.HORN)
        return ClassResult(label=label, confidence=confidence, is_emergency=emergency)


# ---------------------------------------------------------------------------
# ② 방향 추정 모듈 출력  (팀원 담당 — 인터페이스만 정의)
# ---------------------------------------------------------------------------
class Direction(str, Enum):
    FRONT = "front"
    REAR = "rear"
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class DirectionResult:
    direction: Direction
    angle_deg: Optional[float] = None       # 원시 각도(있으면). 없으면 None.


# ---------------------------------------------------------------------------
# ③ 접근/멀어짐 모듈 출력  (팀원 담당 — 인터페이스만 정의)
# ---------------------------------------------------------------------------
class Motion(str, Enum):
    APPROACHING = "approaching"
    RECEDING = "receding"
    STEADY = "steady"
    UNKNOWN = "unknown"


@dataclass
class ApproachResult:
    motion: Motion


# ---------------------------------------------------------------------------
# 최종 통합 결과
# ---------------------------------------------------------------------------
@dataclass
class FusedResult:
    """pipeline 이 세 모듈 결과를 합친 최종 출력."""
    sound: ClassResult
    direction: DirectionResult
    approach: ApproachResult

    def to_korean(self) -> str:
        """예: '사이렌, 후방, 접근 중'"""
        ko_sound = {
            SoundClass.SIREN: "사이렌",
            SoundClass.HORN: "경적",
            SoundClass.NORMAL_TRAFFIC: "일반 도로 소음",
        }[self.sound.label]
        ko_dir = {
            Direction.FRONT: "전방",
            Direction.REAR: "후방",
            Direction.LEFT: "좌측",
            Direction.RIGHT: "우측",
            Direction.UNKNOWN: "방향 미상",
        }[self.direction.direction]
        ko_motion = {
            Motion.APPROACHING: "접근 중",
            Motion.RECEDING: "멀어짐",
            Motion.STEADY: "유지",
            Motion.UNKNOWN: "이동 미상",
        }[self.approach.motion]
        return f"{ko_sound}, {ko_dir}, {ko_motion}"

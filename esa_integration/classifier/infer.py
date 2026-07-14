"""
① 소리 분류 — 메인 추론 (팀 계약 진입점).

    classifier.infer(chunk: AudioChunk) -> ClassResult      # 팀 약속 시그니처

내부는 Airacle의 학습된 검출기(ViT-CNN-Attention 계열 `CNNAttn`, 3-클래스
{siren, horn, noise})를 벤더링해 감싼다. 팀 계약(16kHz·1초 stateless 청크)과
모델 요구(22.05kHz·5초 윈도우) 사이의 간극은 **이 어댑터가 흡수**한다:

  · 리샘플   16kHz(또는 청크가 준 sr) → 22.05kHz
  · 링버퍼   1초 청크를 누적해 최근 5초 창 유지(모델은 5초 문맥 필요)
  · 정규화·forward → softmax → ClassResult 매핑

⚠ 상태 보유: `infer()`는 모듈 전역 인스턴스에 링버퍼를 유지한다(연속 청크 전제).
   파일 A와 파일 B를 번갈아 넣는 등 스트림이 섞이면 `reset()` 하거나
   `SirenClassifier`를 따로 만들어 쓸 것.

⚠ 이건 '분류' 모듈이다 — 창 하나의 raw 분류만 반환한다. 실서비스 경보 판정
   (로짓 마진 임계 τ, 안정화 게이트, 예비경보)은 Airacle 런타임 쪽 로직이며
   여기(분류 계약)엔 포함하지 않는다.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import torch

from core.types import AudioChunk, ClassResult, SoundClass
from . import preprocessing as pp
from .detector_model import build

_CKPT = Path(__file__).resolve().parent / "weights" / "cnn_attn_full_s42.pt"
_MODEL_NAME = "cnn_attn"                 # 가중치 = CNNAttn(full 증강, s42) — Airacle 현역 검출기

# 모델 3-클래스 → 팀 계약 SoundClass. noise ≈ 일반 도로 소음.
_LABEL_MAP = {
    "siren": SoundClass.SIREN,
    "horn": SoundClass.HORN,
    "noise": SoundClass.NORMAL_TRAFFIC,
}


def _resample(y: np.ndarray, src_sr: int, dst_sr: int = pp.SR) -> np.ndarray:
    """src_sr → dst_sr. scipy 있으면 폴리페이즈(고품질), 없으면 선형 보간 폴백."""
    if src_sr == dst_sr:
        return y.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        g = gcd(src_sr, dst_sr)
        return resample_poly(y, dst_sr // g, src_sr // g).astype(np.float32)
    except Exception:                    # scipy 미설치 — 저품질 폴백(정확도 민감하면 설치 권장)
        dur = len(y) / src_sr
        n = int(round(dur * dst_sr))
        return np.interp(np.linspace(0.0, dur, n, endpoint=False),
                         np.linspace(0.0, dur, len(y), endpoint=False), y).astype(np.float32)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """(n,) 그대로 / (n, ch) → ch0(ReSpeaker 처리 채널). 분류는 단일 채널."""
    return samples if samples.ndim == 1 else samples[:, 0]


class SirenClassifier:
    """연속 청크를 받아 매번 최근 5초 창을 분류(링버퍼 은닉).

    device: 'cpu'(팀 노트북 개발 기본) / 'cuda'(Jetson) / 'mps'(M-mac).
    """

    _WIN_SAMPLES = int(pp.WIN_S * pp.SR)     # 110250 = 최근 5초 @ 22050

    def __init__(self, ckpt: str | Path = _CKPT, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = build(_MODEL_NAME).to(self.device).eval()
        state = torch.load(str(ckpt), map_location=self.device)
        self.model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        self._buf = np.zeros(0, np.float32)

    def reset(self) -> None:
        """스트림 경계(다른 오디오 소스로 전환) 시 링버퍼 비우기."""
        self._buf = np.zeros(0, np.float32)

    @torch.no_grad()
    def infer(self, chunk: AudioChunk) -> ClassResult:
        y = _resample(_to_mono(np.asarray(chunk.samples, np.float32)), chunk.sample_rate)
        self._buf = np.concatenate([self._buf, y])[-self._WIN_SAMPLES:]   # 최근 5초만 유지
        mel = pp.to_model_frames(pp.logmel(self._buf))                    # (64, 216)
        x = torch.from_numpy(np.ascontiguousarray(pp.normalize(mel)))[None, None].to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        i = int(probs.argmax())
        return ClassResult.from_label(_LABEL_MAP[pp.CLASSES[i]], float(probs[i]))


# ── 팀 계약 진입점: 모듈 전역 인스턴스(링버퍼 은닉) ───────────────────────────
_SINGLETON: SirenClassifier | None = None


def infer(chunk: AudioChunk) -> ClassResult:
    """AudioChunk → ClassResult. 연속 호출 간 최근 5초를 내부 유지(stateful).
    스트림을 바꾸면 `reset()`."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SirenClassifier()
    return _SINGLETON.infer(chunk)


def reset() -> None:
    """전역 분류기 링버퍼 초기화."""
    if _SINGLETON is not None:
        _SINGLETON.reset()

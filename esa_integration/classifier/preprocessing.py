"""
분류 전처리 — 오디오 파형 → 로그멜 텐서.

⚠ 이 파일은 Airacle(ViT-CNN-Attention 검출기) 학습 파이프라인의 `dataset.py`에서
**추론 경로만 그대로** 옮긴 것이다. 상수·멜 필터뱅크·logmel 수식을 학습과 1:1로
유지해야 train/infer skew가 없다 — 임의로 바꾸지 말 것(바꾸면 벤더링한 가중치가 깨짐).

출력 규격(팀 계약과 별개인 '모델 입력' 규격, docs/06 §0):
  로그멜 (64, 216), 윈도우별 정규화 후 (1, 1, 64, 216)로 모델에 투입.
"""
from __future__ import annotations

import numpy as np

# ── 모델 입력 사양 (Airacle dataset.py와 동일해야 함) ────────────────────────
SR = 22050                       # 검출기 학습 샘플레이트(팀 계약 16k와 다름 → infer.py가 리샘플)
WIN_S = 5.0                      # 검출 윈도우 길이(초). 팀 계약은 1s 청크 → 어댑터가 링버퍼로 누적
N_MELS, N_FFT, HOP = 64, 1024, 512
N_FRAMES = 1 + int(WIN_S * SR) // HOP     # = 216
LOG_EPS = 1e-6
PAD_VAL = float(np.log(LOG_EPS))          # 5s 미만(워밍업) 꼬리 프레임 무음 패딩값 — 학습과 동일
CLASSES = ("siren", "horn", "noise")      # 모델 출력 3-클래스 순서(dataset.LABEL_IDX)

assert N_FRAMES == 216, "입력 텐서 (1,64,216)과 불일치 — 상수 훼손"


def _mel_fb(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """HTK mel 삼각 필터뱅크 (64, 513)."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    pts = 700.0 * (10.0 ** (np.linspace(mel(0.0), mel(sr / 2), n_mels + 2) / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        c = max(c, l + 1)
        r = max(r, c + 1)
        fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = _mel_fb()
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)   # periodic hann


def logmel(y: np.ndarray) -> np.ndarray:
    """(64, T) 로그멜. center 패딩이라 5 s(=110250 샘플) → T=216."""
    y = np.pad(y, N_FFT // 2, mode="reflect")
    n = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * _WINDOW, axis=1)) ** 2
    return np.log(spec @ _FB.T + LOG_EPS).T.astype(np.float32)


def normalize(m: np.ndarray) -> np.ndarray:
    """윈도우별 정규화 (Airacle infer._norm과 동일)."""
    return ((m - m.mean()) / (m.std() + 1e-5)).astype(np.float32)


def to_model_frames(mel: np.ndarray) -> np.ndarray:
    """가변 길이 로그멜 → 정확히 (64, 216). 5s 미만은 뒤를 무음(PAD_VAL) 패딩,
    초과분은 **가장 최근** 216 프레임 사용(스트리밍 = 최신 5초 창)."""
    if mel.shape[1] >= N_FRAMES:
        return mel[:, -N_FRAMES:]
    return np.pad(mel, ((0, 0), (0, N_FRAMES - mel.shape[1])), constant_values=PAD_VAL)

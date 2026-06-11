"""
augment.py — 멜 도메인 증강 (docs/06 §4)

캐시된 전체-파일 로그멜 위에서 동작 — 파형 재계산 없이 epoch마다 전수 증강.

구현 노트 (근사의 근거):
- 배속(#4)·피치시프트(#2)·타임스트레치(#3)는 멜 축 워프로 구현.
  주파수 스케일 f→k·f 는 HTK mel 중심주파수의 정확한 역매핑으로 보간 (창 누설 차이는 2차 효과).
  시간축은 전체-파일 멜에서 넓게/좁게 슬라이스하므로 윈도우 밖 내용을 잃지 않는다.
- 잡음 합성(#6)은 파워 도메인 혼합 P_mix = P_s + g·P_n — 독립 신호 가정에서 정확
  (E|A+B|² = |A|²+|B|²). SNR은 **이벤트(비패딩) 프레임 전력 기준** (docs/06 §4.1 패딩 정책).
  짧은 클립의 무음 패딩이 자동으로 노이즈필 됨 (P_s≈0 영역 → 순수 배경).
  잡음원은 **같은 split의 noise 클래스만** (누수 채널 차단).
- 게인(#5) ±6 dB는 윈도우별 (x−μ)/σ 정규화에서 항등이라 제외 (로그멜 + c → 불변).
- 슬라이딩(#1)은 dataset.py 청크 그리드가 담당. Mixup/CutMix(#8–9)는 배치 단위라 train.py에.

프리셋 (ablation 사다리, docs/06 §6):
  none      : 증강 없음 (청크 그대로)
  wave      : 워프(배속·피치·스트레치) + 잡음 합성  (≈ 파형 1–6)
  wave_spec : + SpecAugment(#7)
  full      : + Random Erasing(#10)  (Mixup/CutMix는 train.py가 배치단에서 추가)
"""
from __future__ import annotations

import hashlib

import numpy as np

import dataset as D

PRESETS = ("none", "wave", "wave_spec", "full")

# HTK mel 중심 주파수 (dataset._mel_fb와 동일 격자)
_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
_imel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
F_CENTER = _imel(np.linspace(_mel(0.0), _mel(D.SR / 2), D.N_MELS + 2))[1:-1]   # (64,)
LOG_FLOOR = np.float32(np.log(D.LOG_EPS))


def _freq_warp(x: np.ndarray, k: float) -> np.ndarray:
    """주파수 스케일 f→k·f (k>1 = 피치 상승). 멜 축 선형보간, 범위 밖은 가장자리 복제."""
    pos = np.interp(F_CENTER / k, F_CENTER, np.arange(D.N_MELS))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, D.N_MELS - 1)
    w = (pos - lo).astype(np.float32)[:, None]
    return x[lo] * (1 - w) + x[hi] * w


class MelAugment:
    """ChunkDataset transform: Chunk → (64, 216) 증강된 raw 로그멜 (정규화 전).

    deterministic=True 면 청크 키 해시로 (잡음 파일, SNR, 워프) 고정 — val/test 노이즈필용.
    """

    def __init__(self, preset: str, sources, split: str = "train",
                 seed: int = 42, deterministic: bool = False):
        assert preset in PRESETS
        self.preset = preset
        self.det = deterministic
        self.seed = seed
        # 같은 split의 noise 파일만 잡음원으로 (docs/06 §4.1 패딩 정책 — 누수 차단)
        self.noise_files = [(s.wav, s.dur) for s in sources
                            if s.label == "noise" and s.split == split]
        assert self.noise_files, f"split={split}에 noise 파일이 없음"

    # ── 잡음원 ──
    def _noise_window(self, rng) -> np.ndarray:
        wav, dur = self.noise_files[rng.integers(len(self.noise_files))]
        m = D.mel_for_file(wav)
        if m.shape[1] <= D.N_FRAMES:
            return np.pad(m, ((0, 0), (0, D.N_FRAMES - m.shape[1])), constant_values=LOG_FLOOR)
        f0 = int(rng.integers(m.shape[1] - D.N_FRAMES))
        return m[:, f0:f0 + D.N_FRAMES]

    # ── 본체 ──
    def __call__(self, c: D.Chunk) -> np.ndarray:
        if self.det:
            key = f"{self.seed}:{D._nfc(c.wav)}:{c.offset}".encode()
            rng = np.random.default_rng(int(hashlib.sha1(key).hexdigest()[:8], 16))
        else:
            rng = np.random.default_rng(np.random.randint(0, 2 ** 31))

        if self.preset == "none":
            return D.chunk_mel(c)

        # 1) 워프 계수 — 각 증강 독립 적용 p=0.5 (배속 0.9–1.1 · 피치 ±2반음 · 스트레치 0.85–1.15)
        speed = rng.uniform(0.9, 1.1) if rng.random() < 0.5 else 1.0
        pitch = 2.0 ** (rng.uniform(-2, 2) / 12.0) if rng.random() < 0.5 else 1.0
        stretch = rng.uniform(0.85, 1.15) if rng.random() < 0.5 else 1.0
        tf = speed * stretch          # 시간축 소스 배율 (출력 1프레임당 소스 tf프레임)
        ff = speed * pitch            # 주파수 스케일

        # 2) 시간 워프: 전체-파일 멜에서 직접 슬라이스 (윈도우 밖 내용 활용)
        m = D.mel_for_file(c.wav)
        f0 = c.offset * D.SR / D.HOP
        pos = f0 + np.arange(D.N_FRAMES, dtype=np.float64) * tf
        valid = pos <= m.shape[1] - 1                       # 파일 밖 = 패딩(무음)
        pc = np.clip(pos, 0, m.shape[1] - 1)
        lo = np.floor(pc).astype(int)
        hi = np.minimum(lo + 1, m.shape[1] - 1)
        w = (pc - lo).astype(np.float32)[None, :]
        x = m[:, lo] * (1 - w) + m[:, hi] * w
        x[:, ~valid] = LOG_FLOOR

        # 3) 주파수 워프
        if ff != 1.0:
            x = _freq_warp(x, ff)

        # 4) 잡음 합성 (p=0.8) — 단, 패딩이 있으면 항상 (무음은 비실재 입력)
        if rng.random() < 0.8 or not valid.all():
            snr_db = rng.uniform(5.0, 20.0)
            ps = np.exp(x)
            pn = np.exp(self._noise_window(rng))
            sig = ps[:, valid].mean() if valid.any() else ps.mean()
            g = sig / (pn.mean() * 10.0 ** (snr_db / 10.0))
            x = np.log(ps + g * pn + D.LOG_EPS).astype(np.float32)

        # 5) SpecAugment (#7): time ≤20f ×2 · freq ≤8mel ×2, 평균값 채움
        if self.preset in ("wave_spec", "full"):
            fill = x.mean()
            for _ in range(2):
                t = int(rng.integers(1, 21))
                t0_ = int(rng.integers(0, D.N_FRAMES - t))
                x[:, t0_:t0_ + t] = fill
            for _ in range(2):
                f = int(rng.integers(1, 9))
                m0 = int(rng.integers(0, D.N_MELS - f))
                x[m0:m0 + f, :] = fill

        # 6) Random Erasing (#10): p=0.4, 면적 1/8–1/3
        if self.preset == "full" and rng.random() < 0.4:
            area = rng.uniform(1 / 8, 1 / 3) * D.N_MELS * D.N_FRAMES
            ar = rng.uniform(0.5, 2.0)
            h = int(np.clip(np.sqrt(area * ar), 1, D.N_MELS))
            w_ = int(np.clip(np.sqrt(area / ar), 1, D.N_FRAMES))
            r0 = int(rng.integers(0, D.N_MELS - h + 1))
            c0 = int(rng.integers(0, D.N_FRAMES - w_ + 1))
            x[r0:r0 + h, c0:c0 + w_] = x.mean()

        return np.ascontiguousarray(x, dtype=np.float32)


# ── 수치 sanity 체크 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    src = D.split_sources(D.index_sources())
    chunks = D.build_chunks(src, splits=("train",))
    aug = MelAugment("full", src, "train", seed=1)

    # 1) 피치 워프: 960 Hz 부근 빈의 에너지가 +2반음(×1.122)에서 기대 빈으로 이동
    x = np.full((64, 216), LOG_FLOOR, np.float32)
    b960 = int(np.argmin(abs(F_CENTER - 960)))
    x[b960] = 0.0
    y = _freq_warp(x, 2 ** (2 / 12))
    b_exp = int(np.argmin(abs(F_CENTER - 960 * 2 ** (2 / 12))))
    print(f"피치워프: 960Hz bin{b960} → argmax bin{int(y.mean(1).argmax())} (기대 {b_exp})")

    # 2) SNR 정확도: 알려진 신호/잡음 파워로 목표 SNR 재현
    rng = np.random.default_rng(0)
    ps, pn = np.full((64, 216), 4.0), np.full((64, 216), 1.0)
    for snr in (5.0, 20.0):
        g = ps.mean() / (pn.mean() * 10 ** (snr / 10))
        got = 10 * np.log10(ps.mean() / (g * pn.mean()))
        print(f"SNR {snr:.0f}dB → 혼합식 역산 {got:.2f}dB")

    # 3) 짧은 horn: 패딩 프레임이 무음 바닥에서 벗어나는가 (노이즈필)
    short = next(c for c in chunks if c.label == "horn" and c.file_dur < 4)
    raw = D.chunk_mel(short)
    pad0 = int(np.ceil((short.file_dur - short.offset) * D.SR / D.HOP))
    deterministic = MelAugment("wave", src, "train", seed=7, deterministic=True)
    a1, a2 = deterministic(short), deterministic(short)
    print(f"짧은 horn(dur {short.file_dur:.1f}s): 원본 패딩평균 {raw[:, pad0+2:].mean():.1f} → "
          f"노이즈필 {a1[:, pad0+2:].mean():.1f} (바닥 {LOG_FLOOR:.1f})")
    print(f"결정성: 두 호출 동일 = {np.allclose(a1, a2)}")

    # 4) 처리량
    import time
    t = time.perf_counter()
    for c in chunks[:300]:
        aug(c)
    print(f"증강 처리량: {(time.perf_counter()-t)/300*1000:.1f} ms/청크")

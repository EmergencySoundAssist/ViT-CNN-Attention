"""
dataset.py — P0 데이터 파이프라인 (docs/06 §0, §6 P0)

핵심 원칙 (누수 방지): **원본 녹음 단위로 먼저 split → 그 다음 슬라이딩 청크 전개**.
청크 생성 후 random split 하면 같은 원본의 청크가 train/test 양쪽에 들어간다(기존 핸드오프 결함).
  - AI Hub `2.Validation` → test 전용 고정
  - AI Hub `1.Training`  → (클래스, 차종) stratify 후 원본 단위 90/10 train/val
  - 검증: split 간 원본 교집합 = ∅  →  `python dataset.py` 로 실행

데이터 사실 (2026-06 라벨 JSON 전수 스캔):
  - 원천 wav(labelName) = 원본 녹음에서 annotation.area 구간만 잘라낸 클립.
    area.start/end 는 **원본 녹음 좌표**라 wav 내부 오프셋이 아님 → 청크 그리드는 wav 전체에 깐다.
  - 원본:클립 = 1:1 (28,643개), labelName 전역 중복 0 → 클립 단위 split == 원본 단위 split.
  - 44.1 kHz PCM16 (mono/stereo 혼재) → 22.05 kHz mono로 변환.
  - 클립 길이 중앙값: siren 11.1 s / horn 2.8 s / noise 7.6 s
    → siren은 슬라이딩으로 ~10× 자연 증폭, horn은 절반 이상이 5 s 미만(패딩 경로).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from siren_data import DATASET_ROOT, _class_of_path, _nfc, _walk, _wav_index

# ── 공통 입력 사양 (docs/06 §0) ──────────────────────────────────────────
SR = 22050
WIN_S, STRIDE_S = 5.0, 1.0
N_MELS, N_FFT, HOP = 64, 1024, 512
N_FRAMES = 1 + int(WIN_S * SR) // HOP   # = 216 (center 패딩 기준)
LOG_EPS = 1e-6
SEED, VAL_RATIO = 42, 0.10
CLASSES = ("siren", "horn", "noise")
LABEL_IDX = {c: i for i, c in enumerate(CLASSES)}

CACHE = Path(__file__).resolve().parent / "cache"
SOURCES_JSON = CACHE / "sources.json"   # 원본 인덱스 (wav 헤더 스캔이 비싸서 캐시)
assert N_FRAMES == 216, "docs/06 §0의 입력 텐서 (1, 64, 216)과 불일치"


# ── 인덱스 자료구조 ──────────────────────────────────────────────────────
@dataclass
class Source:
    """클립 wav 1개 (= 원본 녹음 1개). split의 단위."""
    wav: str      # 절대경로
    label: str    # siren / horn / noise
    sub: str      # 구급차/경찰차/소방차/대형차/헬리콥터 …
    orig: str     # 원본 녹음 fileName (audio.fileName) — 누수 검증 키
    dur: float    # 클립 길이(초, wav 헤더 기준)
    acq: str      # 자연적 / 인위적 (acqMethod — A–B 일치율 평가용)
    aihub: str    # AI Hub 폴더: train(1.Training) / test(2.Validation)
    split: str    # 최종: train / val / test


@dataclass
class Chunk:
    """슬라이딩 윈도우 1개. 학습 샘플의 단위."""
    wav: str
    label: str
    sub: str
    split: str
    offset: float     # 윈도우 시작(초, 클립 내부)
    file_dur: float   # 클립 길이 — dur < WIN_S 면 로드 시 패딩


# ── 1) 원본 인덱싱 ───────────────────────────────────────────────────────
def _wav_duration(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def index_sources(rebuild: bool = False) -> list[Source]:
    """라벨 JSON 전수 → Source 목록. wav 헤더 스캔(~30 s)은 cache/sources.json에 저장."""
    if SOURCES_JSON.exists() and not rebuild:
        return [Source(**d) for d in json.loads(SOURCES_JSON.read_text())]

    wavs = _wav_index()
    out: list[Source] = []
    for ndp, dp, f in _walk(DATASET_ROOT, ".json"):
        cls = _class_of_path(ndp)
        if cls is None:
            continue
        try:
            with open(os.path.join(dp, f), encoding="utf-8") as fp:
                m = json.load(fp)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        ann = (m.get("annotations") or [{}])[0]
        wp = wavs.get(_nfc(ann.get("labelName") or ""))
        if not wp:
            continue
        try:
            dur = _wav_duration(wp)
        except (wave.Error, OSError, EOFError):
            a = ann.get("area", {})
            dur = max(0.0, float(a.get("end", 0) or 0) - float(a.get("start", 0) or 0))
        out.append(Source(
            wav=wp, label=cls, sub=_nfc(ann.get("subCategory") or ""),
            orig=_nfc(m.get("audio", {}).get("fileName") or os.path.basename(wp)),
            dur=round(dur, 3),
            acq=_nfc(m.get("environment", {}).get("acqMethod") or ""),
            aihub="train" if "1.Training" in ndp else "test",
            split="",
        ))
    CACHE.mkdir(exist_ok=True)
    SOURCES_JSON.write_text(json.dumps([asdict(s) for s in out], ensure_ascii=False))
    return out


# ── 2) 파일 단위 split ───────────────────────────────────────────────────
def split_sources(sources: list[Source], seed: int = SEED,
                  val_ratio: float = VAL_RATIO) -> list[Source]:
    """2.Validation → test 고정. 1.Training → (label, sub) stratum별 결정적 90/10."""
    strata: dict[tuple, list[Source]] = defaultdict(list)
    for s in sources:
        if s.aihub == "test":
            s.split = "test"
        else:
            strata[(s.label, s.sub)].append(s)
    for key, group in sorted(strata.items()):
        group.sort(key=lambda s: s.orig)            # 결정적 순서 후 시드 셔플
        random.Random(f"{seed}:{key}").shuffle(group)
        n_val = max(1, round(len(group) * val_ratio))
        for i, s in enumerate(group):
            s.split = "val" if i < n_val else "train"
    return sources


def verify_no_leak(sources: list[Source]) -> list[str]:
    """split 간 원본 교집합 = ∅ + AI Hub 경로 일관성. 위반 메시지 목록 반환(비면 통과)."""
    errs = []
    origs = {sp: {s.orig for s in sources if s.split == sp} for sp in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = origs[a] & origs[b]
        if inter:
            errs.append(f"{a}∩{b} 원본 {len(inter)}개 누수: {sorted(inter)[:3]} …")
    for s in sources:
        if s.split == "test" and "2.Validation" not in s.wav:
            errs.append(f"test인데 2.Validation 밖: {s.wav}")
        if s.split in ("train", "val") and "1.Training" not in s.wav:
            errs.append(f"{s.split}인데 1.Training 밖: {s.wav}")
    return errs


# ── 3) 청크 인덱스 ───────────────────────────────────────────────────────
def _grid(dur: float) -> list[float]:
    """5 s 윈도우 · 1 s stride. 꼬리 구간은 마지막 윈도우를 끝에 스냅. 5 s 미만은 단일(패딩)."""
    if dur < WIN_S:
        return [0.0]
    offs = [i * STRIDE_S for i in range(int((dur - WIN_S) / STRIDE_S + 1e-9) + 1)]
    tail = round(dur - WIN_S, 3)
    if tail - offs[-1] > 1e-6:
        offs.append(tail)
    return offs


def build_chunks(sources: list[Source], classes=CLASSES,
                 splits=("train", "val", "test")) -> list[Chunk]:
    return [
        Chunk(s.wav, s.label, s.sub, s.split, off, s.dur)
        for s in sources if s.label in classes and s.split in splits
        for off in _grid(s.dur)
    ]


# ── 4) 오디오 → 로그멜 (numpy/scipy만 사용 — Orin에서도 동일 코드) ────────
def load_wav(path: str, sr: int = SR) -> np.ndarray:
    from scipy.io import wavfile
    fs, y = wavfile.read(path)
    if y.dtype == np.int16:               # PCM16 (전수 확인됨)
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
    else:
        y = y.astype(np.float32)
    if y.ndim == 2:                       # 스테레오 → 모노
        y = y.mean(axis=1)
    if fs != sr:
        from scipy.signal import resample_poly
        g = math.gcd(fs, sr)
        y = resample_poly(y, sr // g, fs // g).astype(np.float32)
    return y


def _mel_fb(sr=SR, n_fft=N_FFT, n_mels=N_MELS) -> np.ndarray:
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
    """(64, T) 로그멜. center 패딩이라 5 s → T=216 (docs/06 §0)."""
    y = np.pad(y, N_FFT // 2, mode="reflect")
    n = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * _WINDOW, axis=1)) ** 2
    return np.log(spec @ _FB.T + LOG_EPS).T.astype(np.float32)


def mel_for_file(wav: str, write_cache: bool = True) -> np.ndarray:
    """클립 전체의 로그멜을 캐시(npy). 청크는 여기서 프레임 슬라이스."""
    key = hashlib.sha1(_nfc(os.path.relpath(wav, DATASET_ROOT)).encode()).hexdigest()
    p = CACHE / "mel" / f"{key}.npy"
    if p.exists():
        return np.load(p)
    m = logmel(load_wav(wav))
    if write_cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, m)
    return m


def chunk_mel(c: Chunk) -> np.ndarray:
    """청크 1개 → (64, 216). 짧은 클립/꼬리는 무음(log eps)으로 패딩."""
    m = mel_for_file(c.wav)
    f0 = int(round(c.offset * SR / HOP))
    w = m[:, f0:f0 + N_FRAMES]
    if w.shape[1] < N_FRAMES:
        w = np.pad(w, ((0, 0), (0, N_FRAMES - w.shape[1])),
                   constant_values=np.float32(np.log(LOG_EPS)))
    return w


# ── 5) torch Dataset (torch는 여기서만 lazy import) ──────────────────────
class ChunkDataset:
    """torch.utils.data.Dataset 호환. transform은 (64,216) np.ndarray → 동일 shape."""

    def __init__(self, chunks: list[Chunk], transform=None):
        self.chunks, self.transform = chunks, transform

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i: int):
        import torch
        c = self.chunks[i]
        x = chunk_mel(c)
        x = (x - x.mean()) / (x.std() + 1e-5)      # 클립별 정규화 (docs/06 §0)
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(np.ascontiguousarray(x))[None], LABEL_IDX[c.label]


def make_weighted_sampler(chunks: list[Chunk], class_ratio=None):
    """배치 균형 샘플러. 기본 비율 siren:horn:noise = 1:1:2 (docs/05)."""
    import torch
    ratio = class_ratio or {"siren": 1.0, "horn": 1.0, "noise": 2.0}
    cnt = Counter(c.label for c in chunks)
    w = [ratio[c.label] / cnt[c.label] for c in chunks]
    return torch.utils.data.WeightedRandomSampler(w, num_samples=len(chunks), replacement=True)


# ── CLI: 인덱스 빌드 + 누수 검증 + 통계 ──────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P0 인덱스 빌드 · 누수 검증 · 멜 캐시")
    ap.add_argument("--rebuild", action="store_true", help="wav 헤더 재스캔 (sources.json 갱신)")
    ap.add_argument("--precache", type=int, nargs="?", const=-1, default=0,
                    metavar="N", help="멜 캐시 선계산 (N개 클립, 생략 시 전체; 재실행 시 이어서)")
    args = ap.parse_args(argv)

    sources = split_sources(index_sources(rebuild=args.rebuild))
    errs = verify_no_leak(sources)
    chunks = build_chunks(sources)

    n_src = Counter((s.split, s.label) for s in sources)
    n_chk = Counter((c.split, c.label) for c in chunks)
    print(f"원본 {len(sources):,}개 → 청크 {len(chunks):,}개  (윈도우 {WIN_S:.0f} s · stride {STRIDE_S:.0f} s)")
    print(f"{'split':6s} {'class':6s} {'원본':>7s} {'청크':>9s}")
    for sp in ("train", "val", "test"):
        for cl in CLASSES:
            print(f"{sp:6s} {cl:6s} {n_src[(sp, cl)]:7,d} {n_chk[(sp, cl)]:9,d}")
    if errs:
        print("\n[누수 검증 실패]")
        for e in errs[:10]:
            print(" ✗", e)
        return 1
    print("\n[누수 검증 통과] train/val/test 원본 교집합 = ∅, AI Hub 경로 일관 ✓")

    if args.precache:
        todo = sorted({c.wav for c in chunks})
        if args.precache > 0:
            todo = todo[: args.precache]
        for i, wav in enumerate(todo, 1):
            mel_for_file(wav)
            if i % 500 == 0 or i == len(todo):
                print(f"  멜 캐시 {i}/{len(todo)}")
    else:  # 스모크 테스트: 클래스별 1청크
        for cl in CLASSES:
            c = next(c for c in chunks if c.label == cl and c.split == "train")
            x = chunk_mel(c)
            print(f"  smoke {cl:6s} shape={x.shape} μ={x.mean():+.2f} σ={x.std():.2f}"
                  f" (dur {c.file_dur:.1f}s off {c.offset:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
AI Hub '도시 소리 데이터' 클립 인덱서 (Airacle)
- macOS 파일명 NFD ↔ JSON NFC 불일치를 전부 NFC로 정규화해 해결
- 지금은 사이렌만 쓰지만 horn/noise 확장이 한 줄(classes=)로 되도록 설계
- 오디오는 로드하지 않고 (경로, 라벨, 차종, 구간, split) 메타만 인덱싱
"""
import os
import json
import unicodedata
from dataclasses import dataclass

DATASET_ROOT = "/Users/swlee/PycharmProjects/Airacle/130.도시 소리 데이터/01.데이터"

# 소분류 폴더 substring → Airacle 3-클래스
LEAF_CLASS = [
    ("2.차량사이렌", "siren"),
    ("1.차량경적", "horn"), ("4.이륜차경적", "horn"),
    ("3.차량주행음", "noise"), ("5.이륜차주행음", "noise"),
    ("6.비행기", "noise"), ("7.헬리콥터", "noise"),
    ("8.기차", "noise"), ("9.지하철", "noise"),
]


@dataclass
class Clip:
    wav: str          # WAV 절대경로 — 파일 자체가 이미 area 구간만 잘린 클립
    label: str        # siren / horn / noise
    sub: str          # 구급차 / 경찰차 / 소방차 등 (없으면 "")
    start: float      # annotation.area 시작(초) ⚠ 원본 녹음 좌표 — wav 내부 오프셋 아님!
    end: float        # annotation.area 끝(초)    (end-start ≈ wav 길이. docs/01 함정 #3)
    split: str        # AI Hub 폴더 기준: train=1.Training / val=2.Validation
                      # 학습용 train/val/test 분할의 정본은 dataset.py (여기 split 아님)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s else s


def _walk(root: str, ext: str):
    for dp, _, files in os.walk(root):
        ndp = _nfc(dp)
        for f in files:
            if f.lower().endswith(ext):
                yield ndp, dp, f


def _class_of_path(nfc_dir: str) -> str | None:
    for leaf, cls in LEAF_CLASS:
        if leaf in nfc_dir:
            return cls
    return None


def _wav_index() -> dict:
    """NFC 기준 basename → 실제 경로."""
    idx = {}
    for _, dp, f in _walk(DATASET_ROOT, ".wav"):
        idx.setdefault(_nfc(f), os.path.join(dp, f))
    return idx


def index_clips(classes=("siren",), splits=("train", "val")) -> list[Clip]:
    """
    원하는 클래스/split 의 Clip 리스트 반환.
    classes=None 이면 전체(siren/horn/noise).
    """
    want = set(classes) if classes else None
    wavs = _wav_index()
    out: list[Clip] = []

    for ndp, dp, f in _walk(DATASET_ROOT, ".json"):
        cls = _class_of_path(ndp)
        if cls is None or (want is not None and cls not in want):
            continue
        split = "train" if "1.Training" in ndp else "val"
        if split not in splits:
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
        a = ann.get("area", {})
        out.append(Clip(
            wav=wp, label=cls, sub=(ann.get("subCategory") or ""),
            start=float(a.get("start", 0) or 0),
            end=float(a.get("end", 0) or 0), split=split,
        ))
    return out


if __name__ == "__main__":
    from collections import Counter
    clips = index_clips(classes=("siren",))
    print(f"사이렌 클립: {len(clips)}")
    print("  split :", dict(Counter(c.split for c in clips)))
    print("  차종  :", dict(Counter(c.sub for c in clips)))
    print("  예시  :", clips[0])

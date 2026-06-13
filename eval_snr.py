"""
eval_snr.py — 저SNR 견고성 스트레스 평가 (재학습 없이 체크포인트 추론만)

클린 test가 천장(0.997)에 붙어 모델 차이가 안 보이므로, 진짜 차별점인 **저SNR 사이렌 recall**을 측정.
test 청크에 test-split noise를 SNR {clean, 20, 10, 5, 0 dB}로 합성 → 모델별:
  - siren recall  (놓친 사이렌 = 안전사고 — 가장 중요)
  - FA/hour       (운용점 τ는 클린 val에서 고정, docs/06 §5.3)
  - macro-F1

잡음 합성은 augment의 파워 도메인 혼합 재사용, **결정적**(청크·SNR 해시) — 모델 간 동일 조건.

사용:
  python eval_snr.py                       # models/*.pt 자동 발견, seed 42
  python eval_snr.py --models cnn_attn_full vit_full --limit 4000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import dataset as D
import models as M
from train import confusion, prf, operating_tau

SNRS = [("clean", None), ("20dB", 20.0), ("10dB", 10.0), ("5dB", 5.0), ("0dB", 0.0)]
RESULTS = Path(__file__).resolve().parent / "results"


class SNRNoise:
    """test 청크에 test-split noise를 고정 SNR로 합성 (결정적). snr_db=None이면 원본."""

    def __init__(self, sources, snr_db, split="test"):
        self.snr_db = snr_db
        self.noise = [(s.wav, s.dur) for s in sources
                      if s.label == "noise" and s.split == split]

    def __call__(self, c: D.Chunk) -> np.ndarray:
        x = D.chunk_mel(c)
        if self.snr_db is None:
            return x
        key = f"{D._nfc(c.wav)}:{c.offset}:{self.snr_db}".encode()
        rng = np.random.default_rng(int(hashlib.sha1(key).hexdigest()[:8], 16))
        wav, _ = self.noise[rng.integers(len(self.noise))]
        m = D.mel_for_file(wav)
        if m.shape[1] <= D.N_FRAMES:
            nb = np.pad(m, ((0, 0), (0, D.N_FRAMES - m.shape[1])),
                        constant_values=np.float32(np.log(D.LOG_EPS)))
        else:
            f0 = int(rng.integers(m.shape[1] - D.N_FRAMES))
            nb = m[:, f0:f0 + D.N_FRAMES]
        ps, pn = np.exp(x), np.exp(nb)
        g = ps.mean() / (pn.mean() * 10.0 ** (self.snr_db / 10.0))
        return np.log(ps + g * pn + D.LOG_EPS).astype(np.float32)


@torch.no_grad()
def infer(model, chunks, transform, device, batch=256):
    from torch.utils.data import DataLoader
    ds = D.ChunkDataset(chunks, transform=transform)
    dl = DataLoader(ds, batch_size=batch, num_workers=6, persistent_workers=True)
    probs, ys = [], []
    for x, y in dl:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="체크포인트 stem (생략 시 models/*_s42.pt 전부)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="스모크: test 청크 수 제한")
    args = ap.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    src = D.split_sources(D.index_sources())
    test = D.build_chunks(src, splits=("test",))
    val = D.build_chunks(src, splits=("val",))
    if args.limit:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(test); test = test[:args.limit]

    stems = args.models or sorted(
        p.stem for p in Path("models").glob(f"*_s{args.seed}.pt")
        if p.stem.count("_") >= 2)            # model_aug_s42 형태만

    def parse(stem: str):
        """stem(예: cnn_attn_full_s42) → (model, aug). 모델명에 밑줄 있어 키 매칭으로 분리."""
        body = stem[:stem.rfind("_s")] if "_s" in stem else stem   # 끝 _sNN 제거
        for mn in sorted(M.MODELS, key=len, reverse=True):         # 긴 키 우선 (cnn_attn > cnn)
            if body == mn or body.startswith(mn + "_"):
                return mn, (body[len(mn) + 1:] or "none")
        raise ValueError(f"모델명 파싱 실패: {stem}")

    rows = []
    for stem in stems:
        mname, aug = parse(stem)
        ckpt = Path(f"models/{stem}.pt")
        if not ckpt.exists():                                      # --models가 _sNN 생략한 경우
            ckpt = Path(f"models/{stem}_s{args.seed}.pt")
        model = M.build(mname).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        model.eval()
        # 운용점 τ는 클린 val에서 고정 (docs/06 §5.3) — SNR 스윕 내내 동일 임계
        vp, vy = infer(model, val, SNRNoise(src, None), device)
        tau = operating_tau(vp, vy, 0.95)

        line = {"model": mname, "aug": aug, "tau": round(tau, 4), "snr": {}}
        cells = []
        for tag, snr in SNRS:
            p, y = infer(model, test, SNRNoise(src, snr), device)
            _, rec, _, macro = prf(confusion(y, p.argmax(1)))
            pred_siren = p[:, 0] >= tau
            sr = float(pred_siren[y == 0].mean())               # 운용점 τ 기준
            fa = float(pred_siren[y != 0].mean()) * 3600.0
            line["snr"][tag] = {"siren_recall": round(sr, 4),
                                "siren_recall_argmax": round(float(rec[0]), 4),  # τ 무관
                                "fa_per_hour": round(fa, 1), "macro_f1": round(macro, 4)}
            cells.append(f"{rec[0]:.3f}")
        rows.append(line)
        print(f"{mname:9s} {aug:10s} τ={tau:.3f} | argmax siren recall @ "
              f"{'/'.join(t for t,_ in SNRS)} = {'/'.join(cells)}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "snr_sweep.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    # 요약표 1: argmax siren recall (τ 무관 — 순수 분류 견고성)
    print(f"\n{'='*72}\nargmax siren recall vs SNR (τ 무관, 놓친 사이렌 = 안전사고)\n{'='*72}")
    print(f"{'model/aug':22s} " + " ".join(f"{t:>7s}" for t, _ in SNRS) + "   Δ(clean→0dB)")
    for r in rows:
        c0 = r['snr']['clean']['siren_recall_argmax'] - r['snr']['0dB']['siren_recall_argmax']
        print(f"{r['model']+'/'+r['aug']:22s} " +
              " ".join(f"{r['snr'][t]['siren_recall_argmax']:7.3f}" for t, _ in SNRS) +
              f"   -{c0:.3f}")
    # 요약표 2: 운용점 τ 기준 recall + FA
    print(f"\n운용점 τ(클린 val 95%) 기준 siren recall")
    print(f"{'model/aug':22s} " + " ".join(f"{t:>7s}" for t, _ in SNRS))
    for r in rows:
        print(f"{r['model']+'/'+r['aug']:22s} " +
              " ".join(f"{r['snr'][t]['siren_recall']:7.3f}" for t, _ in SNRS))
    print(f"\nFA/hour vs SNR (낮을수록 좋음)")
    print(f"{'model/aug':22s} " + " ".join(f"{t:>7s}" for t, _ in SNRS))
    for r in rows:
        print(f"{r['model']+'/'+r['aug']:22s} " +
              " ".join(f"{r['snr'][t]['fa_per_hour']:7.0f}" for t, _ in SNRS))


if __name__ == "__main__":
    main()

"""
train.py — P1 검출 사다리 학습·평가 (docs/06 §5)

사용:
  python train.py --model cnn|cnn_attn|vit [--seed 42] [--epochs 50] [--limit N(스모크)]

설정 (docs/06 §5.4):
  AdamW lr 1e-3 wd 1e-4 · cosine (ViT는 warmup 5ep) · batch 64 · 최대 50ep
  early stop: val macro-F1, patience 10
  배치 균형: 1:1:2 WeightedRandomSampler — CE는 무가중 (이중 보정 금지, docs/06 §5.4)
  ViT만 label smoothing 0.1 (DeiT 레시피)

평가:
  - test macro-F1, per-class P/R/F1, 혼동행렬 (argmax)
  - 운용점: val에서 siren recall ≥ 95%가 되는 임계 τ → test에서 siren recall·FA/hour
    (FA/hour = 비사이렌 청크의 siren 오판률 × 3600 — 1 s stride 스트림 가정)
결과: results/ladder.json append + 콘솔 요약
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset as D
import models as M

RESULTS = Path(__file__).resolve().parent / "results"
CKPT_DIR = Path(__file__).resolve().parent / "models"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── 지표 (sklearn 없이) ──────────────────────────────────────────────────
def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> np.ndarray:
    cm = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf(cm: np.ndarray):
    """per-class precision/recall/F1 + macro-F1"""
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1, float(f1.mean())


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def operating_tau(val_probs: np.ndarray, val_ys: np.ndarray, target_recall=0.95) -> float:
    """val에서 siren recall ≥ target이 되는 가장 높은 임계 τ (p_siren ≥ τ → siren)."""
    p_siren = val_probs[val_ys == 0, 0]
    return float(np.quantile(p_siren, 1.0 - target_recall))


def at_tau(probs: np.ndarray, ys: np.ndarray, tau: float):
    pred_siren = probs[:, 0] >= tau
    siren_recall = float(pred_siren[ys == 0].mean()) if (ys == 0).any() else float("nan")
    fa_rate = float(pred_siren[ys != 0].mean())          # 비사이렌 청크의 siren 오판률
    return siren_recall, fa_rate * 3600.0                # → FA/hour (1 s stride)


# ── 학습 ────────────────────────────────────────────────────────────────
def lr_lambda_factory(epochs: int, warmup: int):
    def f(ep):  # 0-indexed epoch
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(M.MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="스모크: split당 청크 수 제한")
    args = ap.parse_args(argv)

    seed_all(args.seed)
    device = pick_device()

    src = D.split_sources(D.index_sources())
    chunks = {sp: D.build_chunks(src, splits=(sp,)) for sp in ("train", "val", "test")}
    if args.limit:
        rng = random.Random(args.seed)
        for sp in chunks:
            rng.shuffle(chunks[sp])
            chunks[sp] = chunks[sp][: args.limit]

    train_ds = D.ChunkDataset(chunks["train"])
    sampler = D.make_weighted_sampler(chunks["train"])          # 1:1:2 (docs/05)
    dl_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, **dl_kw)
    val_dl = DataLoader(D.ChunkDataset(chunks["val"]), batch_size=256, **dl_kw)
    test_dl = DataLoader(D.ChunkDataset(chunks["test"]), batch_size=256, **dl_kw)

    model = M.build(args.model).to(device)
    n_par = M.n_params(model)
    warmup = 5 if args.model == "vit" else 0                    # docs/06 §5.4
    smoothing = 0.1 if args.model == "vit" else 0.0             # DeiT 레시피
    crit = nn.CrossEntropyLoss(label_smoothing=smoothing)       # 무가중 — 샘플러가 균형 담당
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.epochs, warmup))

    print(f"[{args.model}] params {n_par/1e6:.3f}M · device {device.type} · seed {args.seed} · "
          f"train/val/test {len(chunks['train']):,}/{len(chunks['val']):,}/{len(chunks['test']):,}")

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt = CKPT_DIR / f"{args.model}_s{args.seed}.pt"
    best_f1, best_ep, bad = -1.0, -1, 0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        run_loss, n_seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        sched.step()

        vp, vy = predict(model, val_dl, device)
        _, _, _, vf1 = prf(confusion(vy, vp.argmax(1)))
        mark = ""
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_f1": vf1}, ckpt)
            mark = " ←best"
        else:
            bad += 1
        print(f"  ep{ep:02d} loss {run_loss/max(n_seen,1):.4f} val macro-F1 {vf1:.4f}"
              f" lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s){mark}", flush=True)
        if bad >= args.patience:
            print(f"  early stop (patience {args.patience})")
            break

    # ── 최적 체크포인트로 최종 평가 ──
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    vp, vy = predict(model, val_dl, device)
    tp_, ty = predict(model, test_dl, device)

    cm = confusion(ty, tp_.argmax(1))
    prec, rec, f1, macro = prf(cm)
    tau = operating_tau(vp, vy, 0.95)
    op_recall, fa_h = at_tau(tp_, ty, tau)

    res = {
        "model": args.model, "seed": args.seed, "params": n_par,
        "best_epoch": best_ep, "val_macro_f1": round(best_f1, 4),
        "test": {
            "macro_f1": round(macro, 4),
            "per_class": {c: {"precision": round(prec[i], 4), "recall": round(rec[i], 4),
                              "f1": round(f1[i], 4)} for i, c in enumerate(D.CLASSES)},
            "confusion": cm.tolist(),
            "op_point": {"tau": round(tau, 4), "siren_recall": round(op_recall, 4),
                         "fa_per_hour": round(fa_h, 1)},
        },
        "limit": args.limit, "train_sec": round(time.time() - t0),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "ladder.json"
    try:
        hist = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        hist = []
    hist.append(res)
    path.write_text(json.dumps(hist, indent=1, ensure_ascii=False))

    print(f"\n[{args.model} s{args.seed}] test macro-F1 {macro:.4f} | "
          f"siren P/R/F1 {prec[0]:.3f}/{rec[0]:.3f}/{f1[0]:.3f} | "
          f"운용점(τ={tau:.3f}) siren recall {op_recall:.3f}, FA {fa_h:.0f}/h | "
          f"best ep{best_ep} ({res['train_sec']}s)")
    print("혼동행렬 (행=정답 siren/horn/noise):")
    for i, c in enumerate(D.CLASSES):
        print(f"  {c:6s} {cm[i].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

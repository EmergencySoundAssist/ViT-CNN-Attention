"""
export_onnx.py — 검출/속도 모델 ONNX 내보내기 + PyTorch 일치 검증 (배포 #2)

cnn_attn/vit(검출) 또는 speed_neural(속도) → ONNX(고정 batch=1). 내보낸 뒤
onnxruntime로 같은 입력 재추론해 출력별 max|Δ| 확인 → TensorRT 변환 전 일치 보장.

레거시 익스포터(dynamo=False) 사용: torch 2.9의 dynamo 경로는 bidirectional GRU를
잘못 분해해 속도 v가 최대 ~5 km/h 어긋남(2026-06 확인). 레거시는 GRU도 충실(Δ~3e-5).

  $ python export_onnx.py                                 # cnn_attn_full(검출)
  $ python export_onnx.py --ckpt models/speed_neural.pt   # 속도
다음(젯슨): trtexec --onnx=models/xxx.onnx --saveEngine=xxx.trt --fp16

⚠ 속도 모델은 export가 충실해도 실주행 미검증(정지에 ~10 km/h 환각). 검증 전 배포 금지.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import infer   # load_model / 전처리 일관 재사용


def load_any(ckpt: str, model_arg: str | None, device):
    """검출/속도/차종 모델 로드 → (model, name, output_names). 파일명으로 분기."""
    base = os.path.basename(ckpt)
    if base.startswith("speed"):
        import speed_neural as sn
        ck = torch.load(ckpt, map_location=device)
        if ck.get("Ln") != 216:
            raise SystemExit(f"fine 모델(Ln={ck.get('Ln')}) export 미지원 — dummy가 (1,1,64,216) 고정")
        has_dir = bool(ck.get("dir", False))
        m = sn.NeuralSpeed(ck["Ln"], dir_head=has_dir).to(device).eval()
        m.load_state_dict(ck["model"])
        return m, "speed_neural", (["speed", "f0", "dir"] if has_dir else ["speed", "f0"])
    if base.startswith("subtype"):                       # 차종(CNNAttn 3-클래스, raw state_dict)
        import models
        rest = base.replace("subtype_", "")
        name = model_arg or next((n for n in ("cnn_attn", "vit", "cnn") if rest.startswith(n)), "cnn_attn")
        m = models.build(name).to(device).eval()
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        return m, f"subtype_{name}", ["logits"]
    m, name = infer.load_model(ckpt, model_arg, device)
    return m, name, ["logits"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="검출/속도 ONNX 내보내기 + 일치 검증")
    ap.add_argument("--ckpt", default="models/cnn_attn_full_s42.pt", help="체크포인트")
    ap.add_argument("--model", default=None, help="검출 모델명 강제 (기본: 파일명 추론)")
    ap.add_argument("--out", default=None, help="출력 .onnx (기본: ckpt와 동명)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset (TensorRT 호환 17)")
    args = ap.parse_args(argv)

    device = torch.device("cpu")                         # export는 CPU
    model, name, onames = load_any(args.ckpt, args.model, device)
    out = args.out or os.path.splitext(args.ckpt)[0] + ".onnx"

    dummy = torch.randn(1, 1, 64, 216)                   # (1,1,64,216) 로그멜
    torch.onnx.export(model, dummy, out, input_names=["mel"], output_names=onames,
                      opset_version=args.opset, dynamo=False)   # 레거시: GRU 충실
    print(f"[내보냄] {name} → {out}  outputs={onames} (고정 batch=1)")

    # 검증: onnxruntime vs PyTorch 출력별 일치
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    ok = True
    for _ in range(4):
        x = np.random.randn(1, 1, 64, 216).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x))
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = sess.run(None, {"mel": x})
        for nm, r, g in zip(onames, ref, got):
            d = float(np.abs(r.numpy() - g).max())
            ok &= d < 1e-3
            print(f"    {nm:7s} max|Δ|={d:.2e}  {'OK' if d < 1e-3 else '⚠ 불일치'}")
    if name == "speed_neural":
        print("  ⚠ export는 충실하나 모델은 실주행 미검증(정지 환각) — 검증 전 배포 금지")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

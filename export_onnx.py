"""
export_onnx.py — NeuralSpeed 체크포인트 → ONNX (Jetson TensorRT 변환용)

사용:
  python export_onnx.py --ckpt models/speed_neural.pt --out speed_neural.onnx
그 다음 Jetson에서:
  trtexec --onnx=speed_neural.onnx --fp16 --saveEngine=speed_neural.engine
"""
from __future__ import annotations

import argparse

import torch

from speed_neural import NeuralSpeed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="speed_neural.onnx")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    Ln = ck["Ln"]
    model = NeuralSpeed(Ln)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.randn(1, 1, 64, Ln)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["mel"], output_names=["v", "f0"],
        opset_version=17, do_constant_folding=True,
    )
    print(f"ONNX 저장: {args.out}  (입력 1×1×64×{Ln})")
    print("Jetson: trtexec --onnx=%s --fp16 --saveEngine=speed_neural.engine" % args.out)


if __name__ == "__main__":
    raise SystemExit(main())

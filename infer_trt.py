"""
infer_trt.py — TensorRT 통합 추론 런타임 (Jetson 배포)

검출(필수) + 차종(siren 게이트일 때만, 잠정·기본 OFF) → 알림 상태기계(alert.py) → Sink.
판정은 softmax 아니라 **로짓 마진** z[cls]-max(나머지). 속도는 제외(코드 자리만).
멜 1 FFT/tick 공유, 검출 짧은 창·차종 5초 전체 창 각자 정규화.

  $ python3 infer_trt.py --wav clip.wav                      # 파일: raw 검출(마진)
  $ python3 infer_trt.py --live                              # 마이크: 알림 상태기계
  $ python3 infer_trt.py --live --det-window 2 --output console
  $ python3 infer_trt.py --live --subtype-engine models/subtype_cnn_attn_dom_s42.trt  # 차종 잠정
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

import alert
import dataset as ds

CLASSES = ds.CLASSES
PAD = float(np.log(ds.LOG_EPS))
SUBS = ("구급차", "경찰차", "소방차")          # subtype_clf.SUBS 순서


def _chk(ret):
    err = ret[0] if isinstance(ret, tuple) else ret
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
    if isinstance(ret, tuple) and len(ret) > 1:
        return ret[1] if len(ret) == 2 else ret[1:]
    return None


def raw_mel(y: np.ndarray) -> np.ndarray:
    """22.05kHz 모노 → **미정규화** 로그멜 (64,216). 정규화는 _norm에서(창별 분리 위해)."""
    m = ds.logmel(y)
    if m.shape[1] < ds.N_FRAMES:
        m = np.pad(m, ((0, 0), (0, ds.N_FRAMES - m.shape[1])), constant_values=PAD)
    return m[:, :ds.N_FRAMES].astype(np.float32)


def _norm(m: np.ndarray) -> np.ndarray:
    return ((m - m.mean()) / (m.std() + 1e-5)).astype(np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    p = np.exp(z - z.max())
    return p / p.sum()


def frames_for(sec: float) -> int:
    return 1 + int(sec * ds.SR) // ds.HOP


def subtype_label(probs: np.ndarray, conf: float) -> str:
    """차종 확률[3] → 라벨. 최고<conf면 '긴급차량'(경찰↔구급 혼동 회피). tier=ood."""
    i = int(probs.argmax())
    p = float(probs[i])
    return (SUBS[i] if p >= conf else "긴급차량") + f"({p:.2f})"


class TRTModel:
    def __init__(self, path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"엔진 로드 실패: {path}")
        self.ctx = self.engine.create_execution_context()
        self.stream = _chk(cudart.cudaStreamCreate())
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dptr = _chk(cudart.cudaMalloc(nbytes))
            is_in = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io[name] = dict(shape=shape, dtype=dtype, nbytes=nbytes, dptr=dptr, is_in=is_in)
            self.ctx.set_tensor_address(name, int(dptr))
        self.in_name = next(n for n, t in self.io.items() if t["is_in"])
        self.out_name = next(n for n, t in self.io.items() if not t["is_in"])

    def __call__(self, x: np.ndarray) -> dict:
        """입력 → {출력이름: array} (다중 출력 지원: 속도 엔진 speed+f0)."""
        t = self.io[self.in_name]
        host = np.ascontiguousarray(x, dtype=t["dtype"])
        _chk(cudart.cudaMemcpyAsync(int(t["dptr"]), host.ctypes.data, t["nbytes"],
             cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.ctx.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 실패")
        outs = {}
        for name, o in self.io.items():
            if o["is_in"]:
                continue
            arr = np.empty(o["shape"], dtype=o["dtype"])
            _chk(cudart.cudaMemcpyAsync(arr.ctypes.data, int(o["dptr"]), o["nbytes"],
                 cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
            outs[name] = arr
        _chk(cudart.cudaStreamSynchronize(self.stream))
        return outs

    def logits(self, x: np.ndarray) -> np.ndarray:
        """단일 분류 출력 → (C,) 1D."""
        return self(x)[self.out_name].reshape(-1)

    def scalar(self, x: np.ndarray) -> float:
        """크기 1 출력(속도 v) → float. (f0 같은 벡터 출력은 무시.)"""
        return float(next(a for a in self(x).values() if a.size == 1).reshape(-1)[0])


class UnifiedRuntime:
    """검출(마진→상태기계) + 차종(siren 게이트시, 잠정). 속도 제외. 멜 1FFT 공유."""

    def __init__(self, det_engine, subtype_engine=None, speed_engine=None,
                 det_window=None, conf=0.6, dt=0.5):
        self.det = TRTModel(det_engine)
        self.subtype = TRTModel(subtype_engine) if subtype_engine else None
        self.speed = TRTModel(speed_engine) if speed_engine else None   # 디버그/확인용(미검증)
        self.det_frames = frames_for(det_window) if det_window else None
        self.conf = conf
        self.g_siren = alert.Gate(alert.CFG["siren"], dt)
        self.g_horn = alert.Gate(alert.CFG["horn"], dt)

    def step(self, y: np.ndarray):
        raw = raw_mel(y)                                          # FFT 1회
        md = raw if not self.det_frames else raw[:, -self.det_frames:]
        z = self.det.logits(_norm(md)[None, None])               # raw logit (softmax 안 함)
        m_siren = float(z[0] - max(z[1], z[2]))
        m_horn = float(z[1] - max(z[0], z[2]))
        sg = self.g_siren.update(m_siren)
        hg = self.g_horn.update(m_horn)

        v_dbg = self.speed.scalar(_norm(raw)[None, None]) if self.speed is not None else None  # 미검증
        risk = alert.speed_tier(v_dbg) if v_dbg is not None else None

        sub = None
        if sg["active"]:                                         # siren > horn 우선
            kind, margin, gate = "siren", m_siren, sg
            if self.subtype is not None:                         # 차종: 마진 게이트일 때만(argmax 아님)
                sp = _softmax(self.subtype.logits(_norm(raw)[None, None]))   # 5초 전체 창
                sub = subtype_label(sp, self.conf)
        elif hg["active"]:
            kind, margin, gate = "horn", m_horn, hg
        else:
            kind, margin = "none", 0.0
            gate = {"onset": False, "remind": False, "clear": sg["clear"] or hg["clear"]}
        return alert.build_event(kind, margin, gate, sub, risk), z, m_siren, m_horn, v_dbg


def live(rt: UnifiedRuntime, sink, stride_s: float, device=None, verbose=False) -> None:
    import math
    import time as _t
    from collections import deque

    import sounddevice as sd
    from scipy.signal import resample_poly

    dev = sd.query_devices(device, kind="input") if device is not None else sd.query_devices(kind="input")
    cap_sr = int(dev["default_samplerate"])
    g = math.gcd(cap_sr, ds.SR)
    buf = deque(maxlen=int(ds.WIN_S * cap_sr))

    def cb(indata, frames, tinfo, status):
        buf.extend(indata[:, 0])

    print(f"마이크: {dev['name']} @ {cap_sr}Hz → {ds.SR}Hz · {stride_s}s tick · Ctrl-C 종료")
    print("(기본: 경보 이벤트[ONSET/REMIND/CLEAR]만. 워밍업 5s 무탐.)"
          + ("\n[--debug] tick마다 상세(pred·margin·v̂) 출력 — v̂은 미검증" if verbose else ""))
    try:
        with sd.InputStream(channels=1, samplerate=cap_sr, callback=cb, device=device):
            while True:
                _t.sleep(stride_s)
                if len(buf) < buf.maxlen:
                    continue
                y = np.array(buf, dtype=np.float32)
                if cap_sr != ds.SR:
                    y = resample_poly(y, ds.SR // g, cap_sr // g).astype(np.float32)
                ev, z, ms, mh, v = rt.step(y)
                sink.emit(ev)
                if verbose:
                    cls = CLASSES[int(np.argmax(z))]
                    vs = f"  v̂={v:5.1f}→{alert.speed_tier(v)}" if v is not None else ""
                    print(f"  [tick] pred={cls:5s}  m_siren={ms:+.2f}{vs} (미검증)", flush=True)
    except KeyboardInterrupt:
        sink.close()
        print("\n종료.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TensorRT 통합 추론 런타임 (.trt) — 파일/라이브+알림")
    ap.add_argument("--engine", default="models/cnn_attn_full_s42.trt", help="검출 엔진")
    ap.add_argument("--subtype-engine", default=None, help="차종 엔진(.trt). 주면 siren시 차종(잠정)")
    ap.add_argument("--speed-engine", default=None, help="속도 엔진(.trt). 디버그 확인용 — tick마다 v̂(미검증, 경보 미사용)")
    ap.add_argument("--wav", help="파일 모드 (raw 검출)")
    ap.add_argument("--live", action="store_true", help="마이크 라이브 (알림 상태기계)")
    ap.add_argument("--det-window", type=float, default=None, help="검출 전용 창(초). 짧으면 onset 빠름")
    ap.add_argument("--output", default="console", help="sink: console / console,gpio")
    ap.add_argument("--stride", type=float, default=0.5, help="tick 간격(초) = 상태기계 dt")
    ap.add_argument("--conf", type=float, default=0.6, help="차종 신뢰 임계(<면 긴급차량)")
    ap.add_argument("--device", default=None, help="입력 장치(이름 일부/인덱스). 미지정시 default(젯슨은 APE=무음 주의)")
    ap.add_argument("--debug", action="store_true", help="tick마다 상세 출력(도배). 기본은 경보 이벤트만")
    args = ap.parse_args(argv)

    if args.live:
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        rt = UnifiedRuntime(args.engine, subtype_engine=args.subtype_engine,
                            speed_engine=args.speed_engine, det_window=args.det_window,
                            conf=args.conf, dt=args.stride)
        live(rt, alert.make_sink(args.output), args.stride, device=device, verbose=args.debug)
        return 0

    if not args.wav:
        ap.error("--wav 또는 --live 필요")
    det = TRTModel(args.engine)
    z = det.logits(_norm(raw_mel(ds.load_wav(args.wav)))[None, None])
    p = _softmax(z)
    m_siren = float(z[0] - max(z[1], z[2]))
    print(f"{args.wav}\n  " + "  ".join(f"{c}={p[i]:.3f}" for i, c in enumerate(CLASSES))
          + f"   → {CLASSES[int(p.argmax())]}  (siren 마진 {m_siren:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

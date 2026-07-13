"""
infer_trt.py — TensorRT 통합 추론 런타임 (Jetson 배포)

검출(필수) + 차종(siren ON일 때만, 잠정·기본 OFF) → 알림 상태기계(alert.py) → Sink.
판정은 softmax 아니라 **로짓 마진** z[cls]-max(나머지). 속도는 제외(코드 자리만).
멜 1 FFT/tick 공유, 검출 짧은 창·차종 5초 전체 창 각자 정규화.
tick 튐 억제: 차종=시간 다수결(SubtypeVote), 속도 tier=중앙값+히스테리시스(SpeedTracker).
거리감: 사이렌 대역(600–1600Hz) 레벨의 배경 대비 ΔdB(ProximityTracker) — 절대거리 아님.
경보 채널은 우선순위 arbiter(확정>예비>경적)로 중재 — ONSET/CLEAR 엣지 항상 짝 보장.

  $ python3 infer_trt.py --wav clip.wav                      # 파일: 전체 슬라이딩 마진 요약
  $ python3 infer_trt.py --live                              # 마이크: 알림 상태기계
  $ python3 infer_trt.py --live --det-window 2 --output console
  $ python3 infer_trt.py --live --subtype-engine models/subtype_cnn_attn_dom_s42.trt  # 차종 잠정
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter

import numpy as np

import alert
import dataset as ds

trt = cudart = None            # 젯슨 전용 — TRTModel 첫 사용 시 lazy import (맥은 .pt 백엔드)


def _lazy_trt():
    global trt, cudart
    if trt is None:
        import tensorrt as _trt
        from cuda.bindings import runtime as _cudart
        trt, cudart = _trt, _cudart

CLASSES = ds.CLASSES
I_SIREN, I_HORN, I_NOISE = (ds.LABEL_IDX[c] for c in ("siren", "horn", "noise"))
PAD = float(np.log(ds.LOG_EPS))
SUBS = ("구급차", "경찰차", "소방차")          # subtype_clf.SUBS 순서

# 사이렌 지배 대역 멜 bin 마스크 — docs/02: 집중대역 700–1200Hz(85%) + 개체 산포(±15%)·도플러(±4%)
_HZ = 700.0 * (10.0 ** (np.linspace(0.0, 2595.0 * np.log10(1.0 + ds.SR / 2 / 700.0),
                                    ds.N_MELS + 2)[1:-1] / 2595.0) - 1.0)
SIREN_BAND = (_HZ >= 600.0) & (_HZ <= 1600.0)


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


LVL_FRAMES = frames_for(0.5)                   # 거리감 레벨: 최근 0.5s 평균


def band_level_db(raw: np.ndarray) -> float:
    """**미정규화** 로그멜(ln 파워) → 사이렌 대역·최근 0.5s 평균 레벨 dB(re full-scale).
    모델 입력은 창별 정규화가 레벨을 버리므로 거리감은 반드시 raw에서 계산. 절대 SPL 아님."""
    return float(10.0 * np.log10(np.exp(raw[SIREN_BAND][:, -LVL_FRAMES:]).mean() + 1e-12))


def subtype_label(probs: np.ndarray, conf: float) -> str:
    """차종 확률[3] → 라벨. 최고<conf면 '긴급차량'(경찰↔구급 혼동 회피). tier=ood."""
    i = int(probs.argmax())
    p = float(probs[i])
    return (SUBS[i] if p >= conf else "긴급차량") + f"({p:.2f})"


class TRTModel:
    def __init__(self, path: str):
        _lazy_trt()
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


class TorchModel:
    """PyTorch 체크포인트(.pt) 백엔드 — TRTModel과 동일 인터페이스(맥 개발/검증용).
    젯슨 없이도 이중 창 PRE·방향 tier까지 같은 UnifiedRuntime으로 실행 가능."""

    def __init__(self, ckpt: str, kind: str):
        import torch
        import infer as inf
        self._torch, self.dev = torch, inf.pick_device()
        if kind == "speed":
            self.m = inf.load_speed(ckpt, self.dev)
        elif kind == "subtype":
            self.m = inf.load_subtype(ckpt, None, self.dev)
        else:
            self.m = inf.load_model(ckpt, None, self.dev)[0]
        self.m.eval()

    def __call__(self, x: np.ndarray) -> dict:
        with self._torch.no_grad():
            out = self.m(self._torch.from_numpy(np.ascontiguousarray(x)).to(self.dev))
        out = out if isinstance(out, tuple) else (out,)
        return {f"o{i}": o.cpu().numpy() for i, o in enumerate(out)}

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self(x)["o0"].reshape(-1)


def load_engine(path: str, kind: str):
    """.trt → TensorRT(젯슨), .pt → PyTorch(맥). 러너 코드는 동일."""
    return TorchModel(path, kind) if path.endswith(".pt") else TRTModel(path)


class UnifiedRuntime:
    """검출(마진→상태기계) + 차종(siren 게이트시, 잠정). 속도 제외. 멜 1FFT 공유."""

    def __init__(self, det_engine, subtype_engine=None, speed_engine=None,
                 det_window=None, conf=0.6, dt=0.5, fast_engine=None):
        self.det = load_engine(det_engine, "det")
        self.subtype = load_engine(subtype_engine, "subtype") if subtype_engine else None
        self.speed = load_engine(speed_engine, "speed") if speed_engine else None   # 디버그/확인용(미검증)
        # 이중 창: fast(2s 예비, ≈2.7s 반응) + det(5s 확정, recall 무손실). 같은 가중치, 창만 다름.
        self.fast = load_engine(fast_engine, "det") if fast_engine else None
        self.fast_frames = (self.fast.io[self.fast.in_name]["shape"][3]      # TRT: 엔진이 창 크기 보유
                            if self.fast is not None and hasattr(self.fast, "io")
                            else (frames_for(1.5) if self.fast else None))   # torch: 1.5초(배포와 동일)
        self.last_dir = None                                                  # 방향 raw 즉시값(표시용)
        self.det_frames = frames_for(det_window) if det_window else None
        self.conf = conf
        self.g_siren = alert.Gate(alert.CFG["siren"], dt)
        self.g_horn = alert.Gate(alert.CFG["horn"], dt)
        self.g_fast = alert.Gate(alert.CFG["siren_fast"], dt) if self.fast else None
        # 시간상수(중앙값창≈2.2s·히스테리시스≈1s·투표창≈6s)를 stride와 무관하게 유지
        self.vtrack = alert.SpeedTracker(n_med=max(3, round(2.25 / dt)) | 1,
                                         k_switch=max(2, round(1.0 / dt)))
        self.svote = alert.SubtypeVote(SUBS, win=max(8, round(6.0 / dt)))
        self.prox = alert.ProximityTracker(dt)   # 거리감(배경 대비 사이렌 대역 ΔdB)
        self.prev_rank = 0                        # arbiter 직전 순위(확정3>예비2>경적1>없음0)

    def step(self, y: np.ndarray):
        raw = raw_mel(y)                                          # FFT 1회
        md = raw if not self.det_frames else raw[:, -self.det_frames:]
        z = self.det.logits(_norm(md)[None, None])               # raw logit (softmax 안 함)
        m_siren = float(z[I_SIREN] - max(z[I_HORN], z[I_NOISE]))
        m_horn = float(z[I_HORN] - max(z[I_SIREN], z[I_NOISE]))
        sg = self.g_siren.update(m_siren)
        hg = self.g_horn.update(m_horn)
        fg, m_fast = None, 0.0
        if self.fast is not None:                                # 예비(짧은 창) 게이트 — 매 tick 갱신
            zf = self.fast.logits(_norm(raw[:, -self.fast_frames:])[None, None])
            m_fast = float(zf[I_SIREN] - max(zf[I_HORN], zf[I_NOISE]))
            fg = self.g_fast.update(m_fast)

        v_dbg, dir_idx = None, None
        if self.speed is not None:                               # 미검증(잠정)
            outs = self.speed(_norm(raw)[None, None])
            v_dbg = float(next(a for a in outs.values() if a.size == 1).reshape(-1)[0])
            d3 = next((a for a in outs.values() if a.size == 3), None)   # 방향 헤드(_dir 엔진)
            if d3 is not None:
                dir_idx = int(d3.reshape(-1).argmax())   # (1,3)/(3,) 모두 클래스 idx 보장
        self.last_dir = dir_idx                          # 상태줄 즉시 표시용(경보 tier는 스무딩)

        # 거리감: 배경은 완전 quiet(모든 게이트 OFF + 마진 음수) tick에서만 EMA 갱신
        quiet = (not sg["active"]) and (not hg["active"]) and not (fg and fg["active"]) \
            and m_siren < 0.0 and m_horn < 0.0
        prox_now = self.prox.update(band_level_db(raw), quiet)

        # 우선순위 arbiter: 확정 siren(3) > 예비(2) > horn(1) > 없음(0).
        # onset=순위 상승(예비→확정 에스컬레이션 포함), clear=0으로 하강.
        # 채널별 gate의 onset/clear를 그대로 쓰면 확정 clear 순간 예비가 살아 있을 때
        # CLEAR 엣지가 유실됐음(구버전 결함) — 순위 전이로 판정해 ONSET/CLEAR 짝 보장.
        rank = 3 if sg["active"] else 2 if (fg and fg["active"]) else 1 if hg["active"] else 0
        onset = rank > self.prev_rank
        clear = rank == 0 and self.prev_rank > 0
        self.prev_rank = rank

        sub, risk, pre, prox = None, None, False, None
        if rank == 3:                                            # 확정 siren
            kind, margin, g0 = "siren", m_siren, sg
            prox = prox_now
            if v_dbg is not None:                                # tier는 스무딩+히스테리시스+방향다수결 경유
                risk = self.vtrack.update(v_dbg, dir_idx)
            if self.subtype is not None:
                if self.g_siren.state == "ON":                   # FALLING(사이렌 꺼진 꼬리) 제외
                    sp = _softmax(self.subtype.logits(_norm(raw)[None, None]))   # 5초 전체 창
                    self.svote.add(sp, self.conf)
                if self.svote.n_seen:
                    # FALLING 중엔 새 투표 없이 직전 다수결 동결 표시(의도).
                    # ON 재진입(같은 경보 지속)은 누적 유지, clear에서만 리셋(의도).
                    sub = self.svote.label()                     # 다수결 라벨(단일 tick 아님)
        elif rank == 2:                                          # 예비: 사이렌 가능성(확정 전)
            kind, margin, g0, pre = "siren", m_fast, fg, True    # 짧은 진동·PRE 표시, 리마인더 없음
            prox = prox_now
        elif rank == 1:
            kind, margin, g0 = "horn", m_horn, hg
        else:
            kind, margin, g0 = "none", 0.0, None
        gate = dict(onset=onset, remind=bool(g0 and g0["remind"]), clear=clear)
        if sg["clear"]:                                          # 확정 해제 → 다음 경보 위해 리셋
            self.vtrack.reset()
            self.svote.reset()
        if clear:
            self.prox.reset_trend()                              # 배경 유지, Δ 추세만 리셋
        return alert.build_event(kind, margin, gate, sub, risk, pre=pre, prox=prox), z, m_siren, m_horn, v_dbg


class _Ring:
    """콜백 스레드용 고정 float32 링버퍼(락 동기화).
    구버전 deque는 파이썬 객체 ~24만개를 매 tick np.array 변환(수십 ms) + GIL 구현
    디테일에 암묵 의존했음. t_last/n_status는 오디오 워치독·유실 감시용."""

    def __init__(self, n: int):
        self.buf = np.zeros(n, np.float32)
        self.n, self.i, self.total = n, 0, 0
        self.lock = threading.Lock()
        self.t_last: float | None = None     # 마지막 콜백 시각(monotonic)
        self.n_status = 0                    # overflow 등 status 플래그 누적

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32).ravel()
        with self.lock:
            k = len(x)
            if k >= self.n:
                self.buf[:] = x[-self.n:]
                self.i = 0
            else:
                j = self.i + k
                if j <= self.n:
                    self.buf[self.i:j] = x
                else:
                    r = self.n - self.i
                    self.buf[self.i:] = x[:r]
                    self.buf[:j - self.n] = x[r:]
                self.i = j % self.n
            self.total += k
            self.t_last = time.monotonic()

    def full(self) -> bool:
        return self.total >= self.n

    def snapshot(self) -> np.ndarray:
        with self.lock:
            return np.concatenate((self.buf[self.i:], self.buf[:self.i]))


def live(rt: UnifiedRuntime, sink, stride_s: float, device=None, verbose=False) -> int:
    import math

    import sounddevice as sd
    from scipy.signal import resample_poly

    dev = sd.query_devices(device, kind="input") if device is not None else sd.query_devices(kind="input")
    cap_sr = int(dev["default_samplerate"])
    g = math.gcd(cap_sr, ds.SR)
    ring = _Ring(int(ds.WIN_S * cap_sr))

    def cb(indata, frames, tinfo, status):
        if status:
            ring.n_status += 1               # overflow 등 — 루프에서 스로틀 경고
        ring.write(indata[:, 0])

    print(f"마이크: {dev['name']} @ {cap_sr}Hz → {ds.SR}Hz · {stride_s}s tick · Ctrl-C 종료")
    print("(연속 실시간 상태줄 + 경보 엣지[ONSET/CLEAR] 영구표시. 워밍업 5s.)"
          + ("\n[--debug] tick마다 상세(pred·margin·v̂) 줄 — v̂은 미검증" if verbose else ""))
    warned = 0
    try:
        with sd.InputStream(channels=1, samplerate=cap_sr, callback=cb, device=device):
            t_start = time.monotonic()
            next_t = t_start
            while True:
                # monotonic 스케줄 — 구버전 sleep(stride)는 처리시간만큼 실효 주기가 늘어짐
                next_t += stride_s
                d = next_t - time.monotonic()
                if d > 0:
                    time.sleep(d)
                else:
                    next_t = time.monotonic()          # 밀림: 폭주 대신 기준 리셋
                # 오디오 워치독: 콜백 무수신(마이크/USB 사망, PulseAudio 0-in 포함)이면
                # 스테일 버퍼로 '멀쩡한 척'하지 않고 비정상 종료 → run.sh가 재시작
                last = ring.t_last if ring.t_last is not None else t_start
                gap = time.monotonic() - last
                if gap > max(1.0, 5 * stride_s):
                    sink.close()
                    print(f"\n[치명] 오디오 입력 {gap:.1f}s 무수신 — 스트림 사망으로 판단, "
                          f"종료(code 3). run.sh가 재시작.", file=sys.stderr)
                    return 3
                if ring.n_status > warned:
                    warned = ring.n_status
                    print(f"\n[경고] 오디오 status 플래그 누적 {warned}회(overflow 등) — 샘플 유실 가능",
                          file=sys.stderr)
                if not ring.full():
                    continue
                y = ring.snapshot()
                if cap_sr != ds.SR:
                    y = resample_poly(y, ds.SR // g, cap_sr // g).astype(np.float32)
                ev, z, ms, mh, v = rt.step(y)
                sink.emit(ev)                                    # 경보 엣지(영구 줄)
                if verbose:
                    cls = CLASSES[int(np.argmax(z))]
                    vs = f"  v̂={v:5.1f}→{alert.speed_tier(v)}" if v is not None else ""
                    print(f"  [tick] pred={cls:5s}  m_siren={ms:+.2f}{vs} (미검증)", flush=True)
                else:
                    st = rt.g_siren.state
                    if st in ("OFF", "RISING") and rt.g_fast is not None \
                            and rt.g_fast.state in ("ON", "FALLING"):
                        st = "PRE"      # 예비만 활성 — 상태줄도 PRE(구버전은 '대기'로 오표시)
                    sink.tick(ms, st, ev.level, ev.risk, rt.last_dir, ev.prox)   # 연속 상태줄
    except KeyboardInterrupt:
        sink.close()
        print("\n종료.")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TensorRT 통합 추론 런타임 (.trt) — 파일/라이브+알림")
    ap.add_argument("--engine", default="models/cnn_attn_full_s42.trt", help="검출 엔진")
    ap.add_argument("--subtype-engine", default=None, help="차종 엔진(.trt). 주면 siren시 차종(잠정)")
    ap.add_argument("--speed-engine", default=None, help="속도 엔진(.trt). 디버그 확인용 — tick마다 v̂(미검증, 경보 미사용)")
    ap.add_argument("--fast-engine", default=None,
                    help="예비검출 엔진(.trt, 2초 창). 주면 확정(5s) 전에 PRE 예비경보(≈2.7s)")
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
                            conf=args.conf, dt=args.stride, fast_engine=args.fast_engine)
        return live(rt, alert.make_sink(args.output), args.stride, device=device, verbose=args.debug)

    if not args.wav:
        ap.error("--wav 또는 --live 필요")
    det = load_engine(args.engine, "det")            # .pt(맥)도 지원 (구버전: TRT 전용 + 첫 5s만 평가)
    mel = ds.logmel(ds.load_wav(args.wav))
    if mel.shape[1] < ds.N_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, ds.N_FRAMES - mel.shape[1])), constant_values=PAD)
    step_f = max(1, round(args.stride * ds.SR / ds.HOP))
    margins, preds = [], []
    for f0 in range(0, mel.shape[1] - ds.N_FRAMES + 1, step_f):
        zz = det.logits(_norm(mel[:, f0:f0 + ds.N_FRAMES].astype(np.float32))[None, None])
        margins.append(float(zz[I_SIREN] - max(zz[I_HORN], zz[I_NOISE])))
        preds.append(int(np.argmax(zz)))
    m = np.array(margins)
    tau = alert.CFG["siren"]["tau_on"]
    print(f"{args.wav}  창 {len(m)}개(5s창·stride {args.stride}s)\n"
          f"  다수결 {CLASSES[Counter(preds).most_common(1)[0][0]]}"
          f"  siren 마진 p50 {np.median(m):+.2f} max {m.max():+.2f}"
          f"  τ_on({tau}) 초과 {int((m >= tau).sum())}창")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

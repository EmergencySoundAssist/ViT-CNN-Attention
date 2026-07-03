"""
infer.py — 검출 추론 런타임 (배포용, docs/04 §실시간 운용)

체크포인트 로드 → WAV → 슬라이딩 5 s 윈도우 → {siren, horn, noise} tick.
전처리(멜·정규화)는 dataset.py를 **그대로 재사용** → 학습/추론 skew 원천 차단.
지금은 파일 슬라이스 입력. 라이브 마이크는 windows()만 교체하면 됨.

  $ python infer.py --wav clip.wav
  $ python infer.py --wav clip.wav --ckpt models/vit_full_s42.pt --stride 0.25
  $ python infer.py --demo            # val 세트에서 클래스별 1클립 자동 추론

출력: tick별 확률 표 + 클립 요약(피크 사이렌 확률 · tick 다수결).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import dataset as ds
import models

CLASSES = ds.CLASSES                       # ("siren", "horn", "noise")
PAD_VAL = float(np.log(ds.LOG_EPS))        # 짧은 꼬리 윈도우 무음 패딩값 (학습과 동일)


def _norm(m: np.ndarray) -> np.ndarray:
    """윈도우별 정규화 (학습/ChunkDataset과 동일)."""
    return ((m - m.mean()) / (m.std() + 1e-5)).astype(np.float32)


def frames_for(sec: float) -> int:
    """초 → 로그멜 프레임 수 (hop 기준)."""
    return 1 + int(sec * ds.SR) // ds.HOP


def pick_device() -> torch.device:
    if torch.cuda.is_available():          # Jetson / 일반 GPU
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # M-series 맥
        return torch.device("mps")
    return torch.device("cpu")


def model_name_from_ckpt(path: str) -> str:
    """파일명 `{model}_{aug}_s{seed}.pt` → 모델명. cnn_attn을 cnn보다 먼저 매칭."""
    base = os.path.basename(path)
    for name in ("cnn_attn", "vit", "cnn"):
        if base.startswith(name):
            return name
    raise ValueError(f"체크포인트 이름에서 모델 추론 실패: {base}")


def load_model(ckpt: str, name: str | None, device) -> tuple[torch.nn.Module, str]:
    name = name or model_name_from_ckpt(ckpt)
    m = models.build(name).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return m, name


def load_speed(ckpt: str, device):
    """속도 신경망 로드. 멜 사양(hop/Ln)이 검출과 같을 때만 멜 공유 가능 → 검증."""
    import speed_neural as sn
    ck = torch.load(ckpt, map_location=device)
    m = sn.NeuralSpeed(ck["Ln"], dir_head=ck.get("dir", False)).to(device).eval()
    m.load_state_dict(ck["model"])
    if (ck["hop"], ck["Ln"]) != (ds.HOP, ds.N_FRAMES):
        raise ValueError(f"속도 멜 사양 {ck['hop']}/{ck['Ln']} ≠ 검출 {ds.HOP}/{ds.N_FRAMES} — 멜 분리 필요")
    return m


SUBS = ["구급차", "경찰차", "소방차"]          # subtype_clf.SUBS 순서


def load_subtype(ckpt: str, name: str | None, device):
    """차종 분류기 로드 — 검출과 동일 CNNAttn 구조(3-클래스), raw state_dict. 멜 공유."""
    name = name or "cnn_attn"
    m = models.build(name).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return m


def subtype_label(probs: np.ndarray, conf: float) -> str:
    """차종 확률[3] → 라벨. 최고확률<conf면 '긴급차량'으로 일반화 (경찰↔구급 혼동 회피)."""
    i = int(probs.argmax())
    p = float(probs[i])
    return (SUBS[i] if p >= conf else "긴급차량") + f"({p:.2f})"


def windows(y: np.ndarray, stride_s: float):
    """전체 로그멜 1회 계산 → stride 간격 (64,216) **미정규화** 윈도우 yield: (start_s, m).
    정규화는 infer_window에서 (검출 짧은 창 분리 위해 raw로 넘김)."""
    mel = ds.logmel(y)                                  # (64, T)
    dur = len(y) / ds.SR
    if dur >= ds.WIN_S:
        n = int((dur - ds.WIN_S) / stride_s + 1e-9) + 1
        starts = [round(i * stride_s, 3) for i in range(n)]
        tail = round(dur - ds.WIN_S, 3)                 # 꼬리 구간 끝에 스냅 (dataset._grid와 동일)
        if tail - starts[-1] > 1e-6:
            starts.append(tail)
    else:
        starts = [0.0]                                  # 5 s 미만 → 단일 윈도우(패딩)
    for t in starts:
        f0 = int(round(t * ds.SR / ds.HOP))
        x = mel[:, f0:f0 + ds.N_FRAMES]
        if x.shape[1] < ds.N_FRAMES:
            x = np.pad(x, ((0, 0), (0, ds.N_FRAMES - x.shape[1])), constant_values=PAD_VAL)
        yield t, x


@torch.no_grad()
def infer_window(m: np.ndarray, det, speed, subtype, device, conf: float = 0.6, det_frames: int | None = None):
    """미정규화 로그멜 (64, N_FRAMES) → (검출 확률[3], 속도 v̂|None, 차종 라벨|None).
    검출은 **마지막 det_frames(짧은 창)** → onset 빠름. 속도·차종은 전체 5초 창(주기·글라이드 필요).
    각 창은 자기 기준으로 정규화. 차종은 검출=siren일 때만."""
    md = m if not det_frames or det_frames >= m.shape[1] else m[:, -det_frames:]
    xd = torch.from_numpy(np.ascontiguousarray(_norm(md)))[None, None].to(device)
    p = torch.softmax(det(xd), dim=1)[0].cpu().numpy()
    v, sub = None, None
    if speed is not None or subtype is not None:
        xf = torch.from_numpy(np.ascontiguousarray(_norm(m)))[None, None].to(device)   # 전체 5초
        if speed is not None:
            v = float(speed(xf)[0].cpu())
        if subtype is not None and CLASSES[int(p.argmax())] == "siren":
            sp = torch.softmax(subtype(xf), dim=1)[0].cpu().numpy()
            sub = subtype_label(sp, conf)
    return p, v, sub


@torch.no_grad()
def infer_file(path: str, det, speed, subtype, device, stride_s: float, conf: float, det_frames=None):
    """파일 1개 → [(tick_start_s, prob[3], v̂, 차종), ...]."""
    y = ds.load_wav(path)
    return [(t, *infer_window(m, det, speed, subtype, device, conf, det_frames))
            for t, m in windows(y, stride_s)]


def report(path: str, ticks) -> None:
    P = np.stack([p for _, p, _, _ in ticks])           # (T, 3)
    has_v = any(t[2] is not None for t in ticks)
    has_s = any(t[3] is not None for t in ticks)
    print(f"\n{os.path.basename(path)}  ({len(ticks)} ticks)")
    print(f"  {'t(s)':>5s} " + " ".join(f"{c:>7s}" for c in CLASSES) + "   →pred"
          + ("   v̂(km/h)" if has_v else "") + ("   차종" if has_s else ""))
    for t, p, v, sub in ticks:
        pred = CLASSES[int(p.argmax())]
        line = f"  {t:5.1f} " + " ".join(f"{x:7.3f}" for x in p) + f"   {pred}"
        if has_v:
            line += f"   {v:6.1f}"
        if has_s:
            line += f"   {sub or '—'}"
        print(line)
    votes = {c: int((P.argmax(1) == i).sum()) for i, c in enumerate(CLASSES)}
    peak = {c: float(P[:, i].max()) for i, c in enumerate(CLASSES)}
    print(f"  요약: 다수결 {max(votes, key=votes.get)} {votes}"
          f" · 피크확률 " + " ".join(f"{c}={peak[c]:.2f}" for c in CLASSES))


def mel_window(y22k: np.ndarray) -> np.ndarray:
    """22.05kHz 모노 윈도우 → **미정규화** 로그멜 (64,216). 정규화는 infer_window에서."""
    m = ds.logmel(y22k)
    if m.shape[1] < ds.N_FRAMES:
        m = np.pad(m, ((0, 0), (0, ds.N_FRAMES - m.shape[1])), constant_values=PAD_VAL)
    return m[:, :ds.N_FRAMES].astype(np.float32)


@torch.no_grad()
def live(det, det_name, speed, subtype, device, stride_s: float, conf: float, det_frames=None) -> None:
    """맥/젯슨 마이크 → 롤링 5s 윈도우 → 검출(+속도+차종) tick. Ctrl-C 종료."""
    import math
    import time as _t
    from collections import deque

    import sounddevice as sd
    from scipy.signal import resample_poly

    dev = sd.query_devices(kind="input")
    cap_sr = int(dev["default_samplerate"])
    g = math.gcd(cap_sr, ds.SR)
    buf = deque(maxlen=int(ds.WIN_S * cap_sr))

    def cb(indata, frames, tinfo, status):
        buf.extend(indata[:, 0])

    det_s = (det_frames * ds.HOP / ds.SR) if det_frames else ds.WIN_S
    print(f"마이크: {dev['name']} @ {cap_sr}Hz → {ds.SR}Hz 리샘플 · 검출창 {det_s:.1f}s / 속도·차종 {ds.WIN_S:.0f}s · {stride_s}s tick")
    print("⚠ 정지 사이렌 v̂≈6~10 정상. 차종은 사이렌일 때만, 애매하면 '긴급차량'.")
    print("사이렌 틀고 Ctrl-C로 종료\n")
    hdr = f"  {'t':>6s} " + " ".join(f"{c:>6s}" for c in CLASSES) + "   pred"
    if speed is not None:
        hdr += "    v̂(km/h)"
    if subtype is not None:
        hdr += "   차종"
    print(hdr)
    try:
        with sd.InputStream(channels=1, samplerate=cap_sr, callback=cb):
            t0 = _t.time()
            while True:
                _t.sleep(stride_s)
                if len(buf) < buf.maxlen:
                    continue
                y = np.array(buf, dtype=np.float32)
                if cap_sr != ds.SR:
                    y = resample_poly(y, ds.SR // g, cap_sr // g).astype(np.float32)
                p, v, sub = infer_window(mel_window(y), det, speed, subtype, device, conf, det_frames)
                pred = CLASSES[int(p.argmax())]
                line = f"  {_t.time()-t0:6.1f} " + " ".join(f"{x:6.3f}" for x in p) + f"   {pred:5s}"
                if v is not None:
                    line += f"  {v:6.1f}"
                if subtype is not None:
                    line += f"   {sub or '—'}"
                print(line)
    except KeyboardInterrupt:
        print("\n종료.")


def demo_clips(n_per_class: int = 1) -> list[str]:
    """val 세트에서 클래스별 클립 경로 (스모크용)."""
    srcs = ds.split_sources(ds.index_sources())
    out = []
    for cl in CLASSES:
        got = [s.wav for s in srcs if s.split == "val" and s.label == cl][:n_per_class]
        out += got
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="검출(+속도) 추론 런타임 — 파일/라이브 마이크")
    ap.add_argument("--wav", help="입력 WAV 경로")
    ap.add_argument("--demo", action="store_true", help="val 세트 클래스별 1클립 자동 추론")
    ap.add_argument("--live", action="store_true", help="마이크 라이브 (맥 내장 마이크 OK)")
    ap.add_argument("--ckpt", default="models/cnn_attn_full_s42.pt", help="검출 체크포인트")
    ap.add_argument("--model", default=None, help="모델명 강제 (기본: 파일명에서 추론)")
    ap.add_argument("--speed", action="store_true", help="속도 신경망도 실행 (v̂ km/h)")
    ap.add_argument("--speed-ckpt", default="models/speed_neural.pt", help="속도 체크포인트")
    ap.add_argument("--subtype", action="store_true", help="차종 분류도 실행 (사이렌일 때 구급/경찰/소방)")
    ap.add_argument("--subtype-ckpt", default="models/subtype_cnn_attn_s42.pt", help="차종 체크포인트")
    ap.add_argument("--subtype-model", default=None, help="차종 모델 구조 (기본 cnn_attn)")
    ap.add_argument("--subtype-conf", type=float, default=0.6,
                    help="이 확률 미만이면 '긴급차량'으로 일반화 (경찰↔구급 혼동 회피)")
    ap.add_argument("--det-window", type=float, default=ds.WIN_S,
                    help=f"검출 전용 창(초). 기본 {ds.WIN_S:.0f}(전체). 줄이면 onset 빠름(예 2). 속도·차종은 항상 5초")
    ap.add_argument("--stride", type=float, default=ds.STRIDE_S, help="tick 간격(초). 라이브 4 Hz면 0.25")
    args = ap.parse_args(argv)

    device = pick_device()
    det, name = load_model(args.ckpt, args.model, device)
    speed = load_speed(args.speed_ckpt, device) if args.speed else None
    subtype = load_subtype(args.subtype_ckpt, args.subtype_model, device) if args.subtype else None
    det_frames = frames_for(args.det_window) if args.det_window < ds.WIN_S else None
    print(f"[검출 {name}] {args.ckpt}"
          + ("  [속도]" if speed else "") + ("  [차종]" if subtype else "")
          + (f"  검출창={args.det_window:.0f}s" if det_frames else "")
          + f"  device={device.type}  stride={args.stride}s")

    if args.live:
        live(det, name, speed, subtype, device, args.stride, args.subtype_conf, det_frames)
        return 0
    paths = [args.wav] if args.wav else demo_clips() if args.demo else None
    if not paths:
        ap.error("--wav · --demo · --live 중 하나는 필요")
    for p in paths:
        report(p, infer_file(p, det, speed, subtype, device, args.stride, args.subtype_conf, det_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

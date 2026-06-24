"""
eval_robust.py — 도메인 증강 효과 측정: 원본 vs _dom, clean vs 채널증강

목적: "채널 증강 학습이 sim-to-real(채널 시프트) 강건성을 올렸나"를 숫자로.
방법: 동일 입력에 clean / 채널증강(학습과 다른 난수 draw = held-out) 둘 다 통과.
  - 차종: val 사이렌 정확도
  - 속도: test 정지풀 합성 passby(알려진 v) 중앙값 오차(km/h)

⚠ 한계: 채널증강은 *시뮬* 도메인 시프트라, 이 평가가 좋아져도 **실주행 전이 보장 아님**.
        진짜 검증은 실녹음. 이건 "증강이 의도대로 작동했나"의 자기일관 측정.

사용: python eval_robust.py
"""
from __future__ import annotations

import random

import numpy as np
import torch

import augment
import dataset as ds
import doppler_speed as dsp
import infer
import speed_head as sh
import speed_neural as sn

dev = infer.pick_device()


@torch.no_grad()
def _cls(model, x_norm):
    xt = torch.from_numpy(np.ascontiguousarray(x_norm))[None, None].to(dev)
    return int(torch.softmax(model(xt), dim=1)[0].argmax())


def eval_subtype(n=120, seed=777):
    sub0 = infer.load_subtype("models/subtype_cnn_attn_s42.pt", None, dev)
    sub1 = infer.load_subtype("models/subtype_cnn_attn_dom_s42.pt", None, dev)
    srcs = ds.split_sources(ds.index_sources())
    vs = [s for s in srcs if s.split == "val" and s.label == "siren" and s.sub in infer.SUBS]
    random.Random(2).shuffle(vs)
    vs = vs[:n]
    rng = np.random.default_rng(seed)                      # held-out 채널 draw
    data = []
    for s in vs:
        y = ds.load_wav(s.wav)
        _, m = next(infer.windows(y, 1.0))
        data.append((infer._norm(m), infer._norm(augment.domain_augment(m, rng)), infer.SUBS.index(s.sub)))

    def acc(model):
        c = sum(_cls(model, cm) == t for cm, _, t in data)
        a = sum(_cls(model, am) == t for _, am, t in data)
        return c / len(data), a / len(data)

    print(f"\n[차종] val 사이렌 {len(data)}개 정확도 — clean / 채널증강(held-out)")
    for tag, mdl in (("원본 ", sub0), ("_dom ", sub1)):
        c, a = acc(mdl)
        print(f"  {tag}  clean {c:.3f}   채널증강 {a:.3f}   (갭 {c-a:+.3f})")


@torch.no_grad()
def eval_speed(n=120, seed=888):
    sp0 = infer.load_speed("models/speed_neural.pt", dev)
    sp1 = infer.load_speed("models/speed_neural_dom.pt", dev)
    pool = sh.stationary_pool("test", 30, 1)
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        seg, sr = pool[rng.integers(len(pool))]
        v = float(rng.uniform(0, 80)); dd = float(rng.uniform(5, 25)); ss = float(rng.uniform(5, 20))
        wc = sh.passby_window(seg, sr, v, dd)
        w = dsp.add_noise(wc, ss, rng)
        data.append((sn.mel_of(w, 512, 216), sn.mel_of_dom(w, 512, 216, rng), v))

    def mae(model, key):
        errs = []
        for clean, aug, v in data:
            x = clean if key == "clean" else aug
            xt = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(dev)
            errs.append(abs(float(model(xt)[0].cpu()) - v))
        return float(np.median(errs))

    print(f"\n[속도] 합성 passby {len(data)}개 중앙값오차(km/h) — clean / 채널증강(held-out)")
    for tag, mdl in (("원본 ", sp0), ("_dom ", sp1)):
        print(f"  {tag}  clean {mae(mdl,'clean'):5.1f}   채널증강 {mae(mdl,'aug'):5.1f}")


def main():
    import os
    if os.path.exists("models/subtype_cnn_attn_dom_s42.pt"):
        eval_subtype()
    else:
        print("[차종] _dom 체크포인트 아직 없음 — subtype 재학습 완료 후")
    if os.path.exists("models/speed_neural_dom.pt"):
        eval_speed()
    else:
        print("[속도] _dom 체크포인트 아직 없음 — speed 재학습 완료 후")


if __name__ == "__main__":
    main()

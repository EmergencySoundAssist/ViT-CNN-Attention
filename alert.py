"""
alert.py — 검출 tick → 안정 알림 상태기계 + 출력 싱크 (순수 파이썬, TRT 무관)

설계: docs 워크플로 스펙. softmax(1.000 포화) 대신 **로짓 마진**으로 판정,
디바운스+히스테리시스+hangover+K/M투표+리마인더. 켜기 쉽게/끄기 느리게(청각장애 안전).
출력은 Sink 인터페이스로 분리(콘솔 지금 / 진동·화면 향후). 속도는 미사용.

⚠ 임계값은 닮은꼴(음악/사이렌FX) hard-negative 평가 전까지 **placeholder**.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass

CLASSES = ("siren", "horn", "noise")
LABEL_KO = {"siren": "사이렌", "horn": "경적", "noise": ""}

# 마진(z[cls]-max(나머지)) 기준 기본 설정 — tick dt=0.5s 가정. ⚠ S0(닮은꼴) 캘리 전 placeholder.
CFG = {
    "siren": dict(tau_on=2.0, tau_off=0.5, N_on=2, T_hang=2.5, K_vote=3, M_win=5, T_remind=3.0),
    "horn":  dict(tau_on=2.5, tau_off=1.0, N_on=2, T_hang=1.0, K_vote=2, M_win=3, T_remind=4.0),
}
TAU_CRIT = 4.0   # 이 마진 이상 + 지속이면 CRITICAL (속도 무관)

# 위험도 tier (속도 v̂ → 정지/접근-느림/접근-빠름). 제품 출력은 km/h가 아니라 이 tier.
# 정지 deadband: v̂이 작으면 무조건 "정지" — OOD 바닥(~10)이 만드는 false-접근 억제.
# ⚠ 경계·deadband는 실주행 캘리 대상(placeholder).
SPEED_TIERS = ("정지", "접근-느림", "접근-빠름")


def speed_tier(v: float, deadband: float = 20.0, fast: float = 40.0) -> str:
    if v < deadband:
        return SPEED_TIERS[0]
    if v < fast:
        return SPEED_TIERS[1]
    return SPEED_TIERS[2]


class Gate:
    """1클래스 디바운스+히스테리시스+hangover+투표+리마인더 상태기계.
    update(margin) → dict(active, onset, remind, clear)."""

    def __init__(self, cfg: dict, dt: float):
        self.cfg, self.dt = cfg, dt
        self.state = "OFF"
        self.run = 0
        self.hang = 0.0
        self.since_remind = 0.0
        self.votes = deque(maxlen=cfg["M_win"])

    def update(self, margin: float) -> dict:
        c = self.cfg
        on_hi = margin >= c["tau_on"]
        on_lo = margin >= c["tau_off"]
        self.votes.append(1 if on_hi else 0)
        voted = sum(self.votes) >= c["K_vote"]
        prev = self.state

        if self.state in ("OFF", "RISING"):
            self.run = self.run + 1 if on_hi else 0
            if self.run >= c["N_on"] and voted:
                self.state = "ON"; self.hang = c["T_hang"]; self.since_remind = 0.0
            else:
                self.state = "RISING" if self.run > 0 else "OFF"
        else:                                            # ON / FALLING
            if on_lo:
                self.state = "ON"; self.hang = c["T_hang"]
            else:
                self.hang -= self.dt
                self.state = "FALLING" if self.hang > 0 else "OFF"
                if self.state == "OFF":
                    self.run = 0

        active = self.state in ("ON", "FALLING")
        onset = active and prev in ("OFF", "RISING")     # 비활성→활성 = 새 경보 (RISING 경유 포함)
        remind = False
        if active:
            self.since_remind += self.dt
            if self.since_remind >= c["T_remind"]:
                remind = True; self.since_remind = 0.0
        clear = (not active) and prev in ("ON", "FALLING")
        return dict(active=active, onset=onset, remind=remind, clear=clear)


@dataclass(frozen=True)
class AlertEvent:
    level: str        # NONE / WARN / CRITICAL
    kind: str         # siren / horn / none
    label: str        # 한글
    margin: float     # 로짓 마진 (softmax 아님)
    onset: bool       # 새 경보 시작
    remind: bool      # ON 지속 리마인더 펄스
    clear: bool       # 해제
    subtype: str | None = None   # "긴급차량(0.71)" tier=ood, 기본 None
    risk: str | None = None      # 위험도 tier(정지/접근-느림/접근-빠름) — 속도엔진 있을 때만


def build_event(kind: str, margin: float, gate: dict | None, subtype: str | None = None,
                risk: str | None = None, tau_crit: float = TAU_CRIT) -> AlertEvent:
    """게이트 판정 → 표시 이벤트. **레벨은 margin 기반**(확률 임계 폐기)."""
    if kind == "siren":
        level = "CRITICAL" if margin >= tau_crit else "WARN"   # 속도 무관, 마진+지속
        return AlertEvent(level, "siren", LABEL_KO["siren"], margin,
                          gate["onset"], gate["remind"], gate["clear"], subtype, risk)
    if kind == "horn":
        return AlertEvent("WARN", "horn", LABEL_KO["horn"], margin,
                          gate["onset"], gate["remind"], gate["clear"], None, None)
    return AlertEvent("NONE", "none", "", 0.0, False, False,
                      gate["clear"] if gate else False, None, None)


# ── 출력 싱크 ──────────────────────────────────────────────────────────────
class ConsoleSink:
    """디버그용. onset/remind/clear 엣지만 출력(매 tick 도배 억제). tty면 색."""
    _COLOR = {"CRITICAL": "\033[1;31m", "WARN": "\033[1;33m", "NONE": "\033[2m"}

    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.t0 = time.time()

    def emit(self, e: AlertEvent) -> None:
        if not (e.onset or e.remind or e.clear):
            return
        tag = "ONSET" if e.onset else ("REMIND" if e.remind else "CLEAR")
        risk = f"  위험도={e.risk}" if e.risk else ""
        sub = f"  차종={e.subtype}(잠정)" if e.subtype else ""
        msg = f"[{time.time()-self.t0:6.1f}] {tag:6s} {e.level:8s} {e.label or '해제':4s}  margin={e.margin:+.2f}{risk}{sub}"
        if self.tty:
            msg = self._COLOR.get(e.level, "") + msg + "\033[0m"
        print(msg, flush=True)

    def close(self):
        pass


class GpioSink:
    """진동(1차)+LED. Jetson.GPIO lazy. ⚠ 미완성 스텁 — 핀 활성화(DTO)·PWM 검증 필요."""
    def __init__(self, vib_pin: int = 33):
        import Jetson.GPIO as GPIO   # noqa: lazy, Jetson 전용
        self.GPIO, self.vib = GPIO, vib_pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(vib_pin, GPIO.OUT, initial=GPIO.LOW)
        self.selftest()

    def selftest(self):
        self.GPIO.output(self.vib, self.GPIO.HIGH); time.sleep(0.3)
        self.GPIO.output(self.vib, self.GPIO.LOW)

    def emit(self, e: AlertEvent) -> None:
        if e.onset or e.remind:
            dur = 0.6 if e.level == "CRITICAL" else 0.3
            self.GPIO.output(self.vib, self.GPIO.HIGH); time.sleep(dur)
            self.GPIO.output(self.vib, self.GPIO.LOW)

    def close(self):
        self.GPIO.cleanup()


class MultiSink:
    def __init__(self, sinks):
        self.sinks = sinks

    def emit(self, e):
        for s in self.sinks:
            try:
                s.emit(e)
            except Exception as ex:                       # 하나 죽어도 나머지 계속
                print(f"[sink 오류] {type(s).__name__}: {ex}", file=sys.stderr)

    def close(self):
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass


def make_sink(spec: str) -> MultiSink:
    """'console' / 'console,gpio' 등 콤마 문자열 → MultiSink."""
    reg = {"console": ConsoleSink, "gpio": GpioSink}
    sinks = []
    for name in (s.strip() for s in spec.split(",") if s.strip()):
        if name not in reg:
            raise ValueError(f"알 수 없는 sink: {name} (가능: {list(reg)})")
        sinks.append(reg[name]())
    return MultiSink(sinks or [ConsoleSink()])


# ── Gate 단위 테스트 (TRT 없이 합성 마진으로) ─────────────────────────────
if __name__ == "__main__":
    dt = 0.5
    g = Gate(CFG["siren"], dt)
    # 마진 시퀀스: 잡음(낮음) → 사이렌(높음) 지속 → 멎음. onset/remind/clear 확인
    seq = [0.1, 0.2, 3.0, 3.5, 3.2, 4.5, 4.0, 3.8, 3.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    print(f"{'t':>4s} {'margin':>7s} {'state':>8s}  events")
    for i, m in enumerate(seq):
        r = g.update(m)
        ev = [k for k in ("onset", "remind", "clear") if r[k]]
        print(f"{i*dt:4.1f} {m:7.2f} {g.state:>8s}  active={int(r['active'])} {ev}")

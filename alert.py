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
from collections import Counter, deque
from dataclasses import dataclass

CLASSES = ("siren", "horn", "noise")
LABEL_KO = {"siren": "사이렌", "horn": "경적", "noise": ""}

# 마진(z[cls]-max(나머지)) 기준 기본 설정 — tick dt=0.5s 가정. ⚠ S0(닮은꼴) 캘리 전 placeholder.
CFG = {
    # τ_on 1.5 근거(2026-07-03 실주행 녹음): 저SNR 실환경(차내, 앞 구급차)서 마진이 0~2에
    # 살아 τ2.0은 25s+ 무경보(첫 경보 34s), τ1.5는 6.2s·커버리지 9→21s. FA 검증: in-domain
    # 비-siren max -3.7, 합성 닮은꼴(알람/음악/처프/노이즈) max -0.01 — 전부 미발화.
    "siren": dict(tau_on=1.5, tau_off=0.5, N_on=2, T_hang=2.5, K_vote=3, M_win=5, T_remind=3.0),
    "horn":  dict(tau_on=2.5, tau_off=1.0, N_on=2, T_hang=1.0, K_vote=2, M_win=3, T_remind=4.0),
    # 예비(PRE) 게이트 — 짧은 창(2s) 검출용: 빨리 켜지고(≈2.7s) 빨리 접음(hangover 1s),
    # 리마인더 없음(확정 채널이 담당). 5s 확정 게이트의 recall은 건드리지 않는다.
    "siren_fast": dict(tau_on=2.0, tau_off=0.5, N_on=2, T_hang=1.0, K_vote=3, M_win=5, T_remind=9999.0),
}
TAU_CRIT = 4.0   # 이 마진 이상 + 지속이면 CRITICAL (속도 무관)

# 위험도 tier. 제품 출력은 km/h가 아니라 tier 문자열.
# SPEED_TIERS = 방향-미상 모드(speed_tier)의 집합. 방향 헤드가 있으면 dir_tier가
# {정지, 멀어짐, 접근-느림, 접근-빠름}을 내며 SpeedTracker는 문자열 비교라 둘 다 수용.
# 정지 deadband: v̂이 작으면 무조건 "정지" — OOD 바닥(~10)이 만드는 false-이동 억제.
# ⚠ "접근/멀어짐"이 아니라 "이동": 현 속도망은 절대속도만 회귀(방향 미학습) —
#   멀어지는 사이렌도 v̂이 크면 잡히므로 접근이라 표기하면 거짓. 방향 헤드(P1) 후 복원.
# ⚠ 경계·deadband는 실주행 캘리 대상(placeholder).
SPEED_TIERS = ("정지", "이동-느림", "이동-빠름")


def speed_tier(v: float, deadband: float = 20.0, fast: float = 40.0) -> str:
    if v < deadband:
        return SPEED_TIERS[0]
    if v < fast:
        return SPEED_TIERS[1]
    return SPEED_TIERS[2]


# 방향 헤드(speed_neural --dir-head) 클래스 순서 = speed_neural.DIR_KO
DIR_KO = ("정지", "접근", "멀어짐")


def dir_tier(dir_idx: int, v: float, fast: float = 40.0) -> str:
    """방향(정지/접근/멀어짐) × 속도 → 위험도 tier. 정지·멀어짐은 방향 헤드가 판정
    (deadband 불필요 — still(v=0) 학습으로 바닥 문제를 원인 치료), 접근만 속도로 세분."""
    if dir_idx == 0:
        return "정지"
    if dir_idx == 2:
        return "멀어짐"
    return "접근-빠름" if v >= fast else "접근-느림"


class SpeedTracker:
    """v̂(+방향) tick 스무딩 + tier 전환 히스테리시스 — 단일 tick 노이즈 깜빡임 억제.
    v는 최근 n_med tick 중앙값, 방향은 다수결. 새 후보 tier가 k_switch tick 연속일 때만 전환.
    ⚠ n_med/k_switch는 tick 수 — 기본값은 0.25s tick 기준(9≈2.2s, 4≈1s).
    stride가 다르면 호출측에서 스케일해 넘길 것(UnifiedRuntime이 dt 기준으로 계산).
    update(v)      → 방향 미상: 정지/이동-느림/이동-빠름 (deadband 기반)
    update(v, dir) → 방향 헤드: 정지/멀어짐/접근-느림/접근-빠름"""

    def __init__(self, n_med: int = 9, k_switch: int = 4):
        self.buf = deque(maxlen=n_med)
        self.dbuf = deque(maxlen=n_med)
        self.k = k_switch
        self.tier = SPEED_TIERS[0]
        self._cand, self._run = None, 0

    def update(self, v: float, dir_idx: int | None = None) -> str:
        self.buf.append(float(v))
        if dir_idx is not None:
            self.dbuf.append(int(dir_idx))
        elif self.dbuf:
            self.dbuf.clear()      # 방향 신호 끊김 → 즉시 속도-only 폴백(오래된 방향 고착 방지)
        s = sorted(self.buf)
        vm = s[len(s) // 2]                          # v 중앙값
        if self.dbuf:
            dm = Counter(self.dbuf).most_common(1)[0][0]   # 방향 다수결
            cand = dir_tier(dm, vm)
        else:
            cand = speed_tier(vm)
        if cand == self.tier:
            self._cand, self._run = None, 0
        elif cand == self._cand:
            self._run += 1
            if self._run >= self.k:
                self.tier, self._cand, self._run = cand, None, 0
        else:
            self._cand, self._run = cand, 1
        return self.tier

    def reset(self) -> None:
        self.buf.clear()
        self.dbuf.clear()
        self.tier = SPEED_TIERS[0]
        self._cand, self._run = None, 0


class SubtypeVote:
    """차종 시간 다수결 — 경보 활성 동안 tick별 argmax를 투표(신뢰 미달 tick은 기권),
    다수 라벨만 표시해 tick간 라벨 튐(경찰↔구급 진동)을 억제. clear 시 reset."""

    def __init__(self, labels=("구급차", "경찰차", "소방차"), win: int = 24):
        self.labels = labels
        self.votes = deque(maxlen=win)               # 클래스 idx만 저장(기권 미저장)
        self.n_seen = 0

    def add(self, probs, conf: float) -> None:
        self.n_seen += 1
        i = int(probs.argmax())
        if float(probs[i]) >= conf:
            self.votes.append(i)

    def label(self) -> str:
        if not self.votes:                           # 유효 투표 0 → 세분화 보류
            return f"긴급차량({self.n_seen}tick)"
        cnt = Counter(self.votes)
        i, c = cnt.most_common(1)[0]
        return f"{self.labels[i]}({c}/{len(self.votes)}표)"

    def reset(self) -> None:
        self.votes.clear()
        self.n_seen = 0


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
                risk: str | None = None, tau_crit: float = TAU_CRIT,
                pre: bool = False) -> AlertEvent:
    """게이트 판정 → 표시 이벤트. **레벨은 margin 기반**(확률 임계 폐기).
    pre=True면 짧은 창 예비경보 — 레벨 PRE 고정(확정 전 단계, 진동 짧게)."""
    if kind == "siren":
        if pre:
            level = "PRE"
        else:
            level = "CRITICAL" if margin >= tau_crit else "WARN"   # 속도 무관, 마진+지속
        return AlertEvent(level, "siren", LABEL_KO["siren"], margin,
                          gate["onset"], gate["remind"], gate["clear"], subtype, risk)
    if kind == "horn":
        return AlertEvent("WARN", "horn", LABEL_KO["horn"], margin,
                          gate["onset"], gate["remind"], gate["clear"], None, None)
    return AlertEvent("NONE", "none", "", 0.0, False, False,
                      gate["clear"] if gate else False, None, None)


# ── 출력 싱크 ──────────────────────────────────────────────────────────────
def _status_line(margin: float, state: str, level: str, risk: str | None = None,
                 dir_raw: int | None = None) -> str:
    """연속 실시간 상태줄(제자리 갱신용). raw 마진 막대 + 안정 게이트 상태.
    막대: 마진 -2(빈칸)~+10(꽉) 12칸, τ_on≈4칸 지점. raw라 매 tick 흔들림(=실시간).
    dir_raw = 방향 헤드의 tick 즉시값(스무딩 전) — 표시만 즉시, 경보 tier는 스무딩 유지."""
    n = max(0, min(12, round((margin + 2.0) / 12.0 * 12)))
    bar = "▓" * n + "░" * (12 - n)
    st = {"OFF": "대기 ", "RISING": "↑감지", "ON": "●경보", "FALLING": "↓유지"}.get(state, state)
    active = state in ("ON", "FALLING")
    lv = f" {level}" if active else ""
    r = f"  위험도={risk}" if (active and risk) else ""
    d = f"  지금={DIR_KO[dir_raw]}" if (active and dir_raw is not None) else ""
    return f"사이렌 [{bar}] {margin:+5.1f}  {st}{lv}{r}{d}"


class ConsoleSink:
    """연속 상태줄(매 tick 제자리 갱신=실시간) + 경보 엣지(ONSET/CLEAR 영구 줄). tty면 색."""
    _COLOR = {"CRITICAL": "\033[1;31m", "WARN": "\033[1;33m", "PRE": "\033[1;33m", "NONE": "\033[2m"}

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
        # 엣지=영구 줄. tty면 진행 중 상태줄을 지우고(\r\033[K) 새 줄에 출력 → 다음 tick이 그 아래에 상태줄.
        print(("\r\033[K" if self.tty else "") + msg, flush=True)

    def tick(self, margin: float, state: str, level: str, risk: str | None = None,
             dir_raw: int | None = None) -> None:
        """매 tick 연속 상태줄을 제자리 갱신 — 실시간 피드백. tty 아니면 생략(파이프 도배 방지)."""
        if not self.tty:
            return
        active = state in ("ON", "FALLING")
        col = self._COLOR.get(level if active else "NONE", "")
        sys.stdout.write("\r\033[K" + col + _status_line(margin, state, level, risk, dir_raw) + "\033[0m")
        sys.stdout.flush()

    def close(self):
        if self.tty:
            sys.stdout.write("\r\033[K"); sys.stdout.flush()   # 잔여 상태줄 정리


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

    def tick(self, *a, **k):                              # 연속 상태줄 — tick 지원 싱크에만 전달
        for s in self.sinks:
            f = getattr(s, "tick", None)
            if f is None:
                continue
            try:
                f(*a, **k)
            except Exception:
                pass

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

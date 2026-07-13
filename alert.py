"""
alert.py — 검출 tick → 안정 알림 상태기계 + 출력 싱크 (순수 파이썬, TRT 무관)

설계: docs 워크플로 스펙. softmax(1.000 포화) 대신 **로짓 마진**으로 판정,
디바운스+히스테리시스+hangover+K/M투표+리마인더. 켜기 쉽게/끄기 느리게(청각장애 안전).
출력은 Sink 인터페이스로 분리(콘솔 지금 / 진동·화면 향후). 속도는 미사용.

⚠ 임계값은 닮은꼴(음악/사이렌FX) hard-negative 평가 전까지 **placeholder**.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

CLASSES = ("siren", "horn", "noise")
LABEL_KO = {"siren": "사이렌", "horn": "경적", "noise": ""}

# 마진(z[cls]-max(나머지)) 기준 기본 설정. 시간상수는 **초 단위** — Gate가 dt(=stride)로
# tick 수 환산하므로 stride를 바꿔도 실효 동작 유지. (기존 tick 단위 N_on/K/M은 stride
# 0.5→0.15 이행 때 디바운스·투표창이 3.3× 조용히 약해지는 결함 — vtrack/svote만 스케일됐었음.)
# 아래 초값은 2026-07 라이브 튜닝 시점(stride 0.15)의 실효 동작과 동일: N_on=2 · K3/M5.
# ⚠ S0(닮은꼴) FA 캘리는 stride 0.5로 측정됐던 것 — eval_hardneg --stride 0.15 재실행 필요.
CFG = {
    # τ_on 1.2 근거(2026-07-06 라이브): 저음압 사이렌이 마진 0.9~1.5에 배회해 1.5 미달로 무경보.
    # (앞서 1.5 근거: 실주행 저SNR서 2.0은 34s 무경보→1.5로 6.2s.) FA 안전마진: in-domain
    # 비-siren max -3.7, 합성 닮은꼴 max -0.01(처프) — 1.2와 갭 충분. 더 내리는 건(1.0↓)
    # 실도로 negative(바람·브레이크음) 데이터 확보 전 금지.
    "siren": dict(tau_on=1.2, tau_off=0.5, t_on=0.3, t_vote=0.75, k_frac=0.6,
                  T_hang=2.5, T_remind=3.0),
    "horn":  dict(tau_on=2.5, tau_off=1.0, t_on=0.3, t_vote=0.45, k_frac=0.67,
                  T_hang=1.0, T_remind=4.0),
    # 예비(PRE) 게이트 — 짧은 창(1.5~2s) 검출용: 빨리 켜지고 빨리 접음(hangover 1s),
    # 리마인더 없음(확정 채널이 담당). 5s 확정 게이트의 recall은 건드리지 않는다.
    "siren_fast": dict(tau_on=2.0, tau_off=0.5, t_on=0.3, t_vote=0.75, k_frac=0.6,
                       T_hang=1.0, T_remind=9999.0),
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


class ProximityTracker:
    """사이렌 대역 레벨(dB re full-scale) → 배경(비경보 EMA) 대비 ΔdB → 거리감 tier + 추세.

    절대 거리·SPL은 불가: 마이크 미캘리브레이션 + 사이렌 원음압 편차(차종·지향성·반사).
    배경 대비 상대 레벨(Δ)만 신뢰 가능 — 마이크 게인이 달라져도 Δ는 불변.
    배경은 quiet(모든 게이트 OFF + 마진 음수) tick에서만 EMA 갱신 → 접근 중 오염 방지.
    ⚠ tier 경계(20/10 dB)는 실주행 캘리 전 placeholder.
    ⚠ 마이크 AGC가 켜져 있으면 레벨이 눌려 무의미 — ReSpeaker AGC OFF 확인 필수.
    ⚠ 사이렌 한복판에서 부팅하면 배경이 오염됨(quiet 복귀 후 T_bg에 걸쳐 자가 회복)."""

    TIERS = ((20.0, "근접"), (10.0, "중간"), (float("-inf"), "원거리"))

    def __init__(self, dt: float, t_bg: float = 30.0, t_med: float = 0.6, t_trend: float = 2.4):
        self.a_bg = min(1.0, dt / t_bg)
        self.buf = deque(maxlen=max(1, round(t_med / dt)))     # Δ 중앙값 스무딩(tick 튐 억제)
        self.tbuf = deque(maxlen=max(4, round(t_trend / dt)))  # 추세(↗/↘) 판정창
        self.bg = None

    def update(self, level_db: float, quiet: bool) -> str | None:
        """매 tick 호출. quiet=배경 갱신 허용. 반환: '근접(+23dB↗)' 꼴(배경 미형성이면 None)."""
        if self.bg is None:
            self.bg = level_db
            return None
        if quiet:
            self.bg += self.a_bg * (level_db - self.bg)
        self.buf.append(level_db - self.bg)
        s = sorted(self.buf)
        med = s[len(s) // 2]
        self.tbuf.append(med)
        arrow = ""
        h = len(self.tbuf) // 2
        if h >= 2:
            older = sorted(list(self.tbuf)[:h])[h // 2]
            newer = sorted(list(self.tbuf)[h:])[(len(self.tbuf) - h) // 2]
            if newer - older >= 2.0:
                arrow = "↗"                                    # 커지는 중(접근 정황)
            elif newer - older <= -2.0:
                arrow = "↘"
        tier = next(t for th, t in self.TIERS if med >= th)
        return f"{tier}({med:+.0f}dB{arrow})"

    def reset_trend(self) -> None:
        """경보 해제 시 — 추세·스무딩만 리셋, 배경(bg)은 유지."""
        self.buf.clear()
        self.tbuf.clear()


class Gate:
    """1클래스 디바운스+히스테리시스+hangover+투표+리마인더 상태기계.
    update(margin) → dict(active, onset, remind, clear).
    t_on/t_vote/k_frac(초·비율)를 dt로 tick 수 환산 — stride 변경에 실효 동작 불변."""

    def __init__(self, cfg: dict, dt: float):
        self.cfg, self.dt = cfg, dt
        self.n_on = max(1, round(cfg["t_on"] / dt))                  # 연속 상회 tick 수
        self.m_win = max(self.n_on, round(cfg["t_vote"] / dt))      # 투표창 tick 수
        self.k_vote = min(self.m_win, max(self.n_on, round(cfg["k_frac"] * self.m_win)))
        self.state = "OFF"
        self.run = 0
        self.hang = 0.0
        self.since_remind = 0.0
        self.votes = deque(maxlen=self.m_win)

    def update(self, margin: float) -> dict:
        c = self.cfg
        on_hi = margin >= c["tau_on"]
        on_lo = margin >= c["tau_off"]
        self.votes.append(1 if on_hi else 0)
        voted = sum(self.votes) >= self.k_vote
        prev = self.state

        if self.state in ("OFF", "RISING"):
            self.run = self.run + 1 if on_hi else 0
            if self.run >= self.n_on and voted:
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
    prox: str | None = None      # 거리감 "근접(+23dB↗)" — 배경 대비 상대 레벨, 절대거리 아님


def build_event(kind: str, margin: float, gate: dict | None, subtype: str | None = None,
                risk: str | None = None, tau_crit: float = TAU_CRIT,
                pre: bool = False, prox: str | None = None) -> AlertEvent:
    """게이트 판정 → 표시 이벤트. **레벨은 margin 기반**(확률 임계 폐기).
    pre=True면 짧은 창 예비경보 — 레벨 PRE 고정(확정 전 단계, 진동 짧게)."""
    if kind == "siren":
        if pre:
            level = "PRE"
        else:
            level = "CRITICAL" if margin >= tau_crit else "WARN"   # 속도 무관, 마진+지속
        return AlertEvent(level, "siren", LABEL_KO["siren"], margin,
                          gate["onset"], gate["remind"], gate["clear"], subtype, risk, prox)
    if kind == "horn":
        return AlertEvent("WARN", "horn", LABEL_KO["horn"], margin,
                          gate["onset"], gate["remind"], gate["clear"], None, None, None)
    return AlertEvent("NONE", "none", "", 0.0, False, False,
                      gate["clear"] if gate else False, None, None, None)


# ── 출력 싱크 ──────────────────────────────────────────────────────────────
def _status_line(margin: float, state: str, level: str, risk: str | None = None,
                 dir_raw: int | None = None, prox: str | None = None) -> str:
    """연속 실시간 상태줄(제자리 갱신용). raw 마진 막대 + 안정 게이트 상태.
    막대: 마진 -2(빈칸)~+10(꽉) 12칸, τ_on≈4칸 지점. raw라 매 tick 흔들림(=실시간).
    dir_raw = 방향 헤드의 tick 즉시값(스무딩 전) — 표시만 즉시, 경보 tier는 스무딩 유지.
    state="PRE" = 예비 게이트만 활성(확정 대기) — 구버전은 '대기'로 오표시했음."""
    n = max(0, min(12, round((margin + 2.0) / 12.0 * 12)))
    bar = "▓" * n + "░" * (12 - n)
    st = {"OFF": "대기 ", "RISING": "↑감지", "ON": "●경보", "FALLING": "↓유지",
          "PRE": "◐예비"}.get(state, state)
    active = state in ("ON", "FALLING", "PRE")
    lv = f" {level}" if active else ""
    r = f"  위험도={risk}" if (active and risk) else ""
    p = f"  거리감≈{prox}" if (active and prox) else ""
    d = f"  지금={DIR_KO[dir_raw]}" if (active and dir_raw is not None) else ""
    return f"사이렌 [{bar}] {margin:+5.1f}  {st}{lv}{r}{p}{d}"


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
        px = f"  거리감≈{e.prox}" if e.prox else ""
        sub = f"  차종={e.subtype}(잠정)" if e.subtype else ""
        msg = f"[{time.time()-self.t0:6.1f}] {tag:6s} {e.level:8s} {e.label or '해제':4s}  margin={e.margin:+.2f}{risk}{px}{sub}"
        if self.tty:
            msg = self._COLOR.get(e.level, "") + msg + "\033[0m"
        # 엣지=영구 줄. tty면 진행 중 상태줄을 지우고(\r\033[K) 새 줄에 출력 → 다음 tick이 그 아래에 상태줄.
        print(("\r\033[K" if self.tty else "") + msg, flush=True)

    def tick(self, margin: float, state: str, level: str, risk: str | None = None,
             dir_raw: int | None = None, prox: str | None = None) -> None:
        """매 tick 연속 상태줄을 제자리 갱신 — 실시간 피드백. tty 아니면 생략(파이프 도배 방지)."""
        if not self.tty:
            return
        active = state in ("ON", "FALLING", "PRE")
        col = self._COLOR.get(level if active else "NONE", "")
        sys.stdout.write("\r\033[K" + col + _status_line(margin, state, level, risk, dir_raw, prox) + "\033[0m")
        sys.stdout.flush()

    def close(self):
        if self.tty:
            sys.stdout.write("\r\033[K"); sys.stdout.flush()   # 잔여 상태줄 정리


class GpioSink:
    """진동(1차)+LED. Jetson.GPIO lazy. ⚠ 미완성 스텁 — 핀 활성화(DTO)·PWM 검증 필요.
    진동 OFF는 Timer로 비동기 — 기존 time.sleep은 stride 0.15 기준 tick 2~4개를 블로킹했음."""
    def __init__(self, vib_pin: int = 33):
        import Jetson.GPIO as GPIO   # noqa: lazy, Jetson 전용
        self.GPIO, self.vib = GPIO, vib_pin
        self._timer: threading.Timer | None = None
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(vib_pin, GPIO.OUT, initial=GPIO.LOW)
        self.selftest()

    def selftest(self):
        self.GPIO.output(self.vib, self.GPIO.HIGH); time.sleep(0.3)
        self.GPIO.output(self.vib, self.GPIO.LOW)

    def emit(self, e: AlertEvent) -> None:
        if e.onset or e.remind:
            dur = 0.6 if e.level == "CRITICAL" else 0.3
            if self._timer is not None:
                self._timer.cancel()                 # 연속 펄스는 뒤 것으로 연장
            self.GPIO.output(self.vib, self.GPIO.HIGH)
            self._timer = threading.Timer(dur, self.GPIO.output, (self.vib, self.GPIO.LOW))
            self._timer.daemon = True
            self._timer.start()

    def close(self):
        if self._timer is not None:
            self._timer.cancel()
        self.GPIO.output(self.vib, self.GPIO.LOW)
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

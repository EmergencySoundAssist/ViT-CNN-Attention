#!/usr/bin/env python3
"""led_display.py — UDP(alert.UdpSink) tick/event 스트림 → 웹 LED바 시각화.

구조(안전 우선):  infer_trt(런타임) ──UDP fire&forget──▶ led_display ──SSE──▶ 브라우저 LED바
  • UDP 수신은 별도 스레드, 최신 스냅샷만 공유(락). 런타임은 디스플레이 생사와 무관.
  • 이 프로세스가 죽어도 경보 런타임은 영향 없음(UdpSink가 조용히 스킵).
  • 표준 라이브러리만 사용 — Jetson 시스템 python(외부 패키지 설치 제약) 대응.

입력 필드(alert.UdpSink):
  tick : margin, state(OFF/RISING/ON/FALLING/PRE), level(NONE/WARN/CRITICAL/PRE),
         risk, dir(정지/접근/멀어짐), prox("근접(+23dB↗)"), prox_db(ΔdB), subtype
  event: tag(onset/remind/clear), level, kind, margin, risk, prox, subtype

LED바 매핑(alert.py 미러):
  감지 강도 = raw margin (-2..+8 → 0..1), 임계선 τ_on=1.2 / τ_crit=4.0 표시.
  거리감    = prox_db (0..30 → 0..1), tier 경계 10(중간)/20(근접), 방향 화살표.
  6.7Hz tick이라도 브라우저가 프레임마다 보간(ease)해 LED바는 부드럽게 움직임.

실행:
  $ python3 led_display.py                 # UDP 127.0.0.1:8737 수신, 웹 0.0.0.0:8080
  $ python3 led_display.py --http-port 8090 --udp-port 8737
그 다음 런타임을 udp sink로:
  $ python3 infer_trt.py --live ... --output console,udp
브라우저에서 http://<젯슨IP>:8080 (같은 기기면 http://localhost:8080).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# alert.py 미러(시각 임계선). 런타임 τ가 바뀌면 여기도 맞춰야 표시가 정확.
TAU_ON = 1.2
TAU_CRIT = 4.0

# ── 공유 최신 상태(UDP 스레드 write, SSE 핸들러 read) ──────────────────────────
_LOCK = threading.Lock()
_STATE: dict = {}            # 최근 tick dict
_SEQ = 0                     # tick 갱신 카운터(SSE가 변경 감지)
_T_RECV = 0.0               # 최근 수신 monotonic(스테일 판정)
_EVENTS: list = []          # 최근 이벤트(onset/remind/clear) 링, 최대 8


def _udp_loop(host: str, port: int) -> None:
    """UDP 데이터그램 수신 → _STATE/_EVENTS 갱신. 손상 패킷은 스킵."""
    global _SEQ, _T_RECV
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    print(f"[led] UDP 수신 {host}:{port}", file=sys.stderr)
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            d = json.loads(data.decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        with _LOCK:
            _T_RECV = time.monotonic()
            if d.get("type") == "event":
                _EVENTS.append(d)
                del _EVENTS[:-8]
            else:                                   # tick(기본)
                _STATE.clear()
                _STATE.update(d)
                _SEQ += 1


def _snapshot() -> str:
    """SSE로 내보낼 현재 상태 JSON. stale=런타임 무수신(디스플레이만 살아있음)."""
    with _LOCK:
        age = time.monotonic() - _T_RECV if _T_RECV else 999.0
        payload = {
            "seq": _SEQ,
            "stale": age > 1.5,                     # 1.5s 무수신이면 '런타임 대기'로 페이드
            "tick": dict(_STATE),
            "events": list(_EVENTS[-4:]),
            "tau_on": TAU_ON, "tau_crit": TAU_CRIT,
        }
    return json.dumps(payload, ensure_ascii=False)


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Airacle — 실시간 사이렌 감지</title>
<style>
  :root{ --bg:#0a0d12; --panel:#12171f; --dim:#2a323d; --txt:#e8edf4; --sub:#8a94a3;
         --green:#3ddc84; --amber:#ffb020; --red:#ff4d4d; }
  *{box-sizing:border-box; margin:0} html,body{height:100%}
  body{background:var(--bg); color:var(--txt); font-family:-apple-system,Segoe UI,Roboto,
       "Noto Sans KR",sans-serif; display:flex; flex-direction:column; min-height:100%;
       padding:clamp(12px,3vw,40px); gap:clamp(12px,2.4vw,28px); overflow:hidden}
  .top{display:flex; align-items:baseline; justify-content:space-between; gap:12px}
  .brand{font-weight:700; letter-spacing:.04em; font-size:clamp(15px,2vw,20px)}
  .brand small{color:var(--sub); font-weight:400; margin-left:.5em}
  #conn{font-size:13px; color:var(--sub)} #conn.on{color:var(--green)}
  /* 상태 배너 */
  .banner{border-radius:16px; padding:clamp(14px,2.6vw,26px) clamp(18px,3vw,32px);
          background:var(--panel); border:1px solid var(--dim); transition:.25s;
          display:flex; align-items:center; gap:clamp(14px,2.4vw,28px)}
  .banner .st{font-size:clamp(30px,7vw,64px); font-weight:800; line-height:1; white-space:nowrap}
  .banner .meta{display:flex; flex-direction:column; gap:6px; min-width:0}
  .banner .lv{font-size:clamp(14px,2vw,20px); font-weight:700; letter-spacing:.05em}
  .banner .row{font-size:clamp(13px,1.7vw,18px); color:var(--sub); display:flex;
               flex-wrap:wrap; gap:6px 18px}
  .banner .row b{color:var(--txt); font-weight:600}
  .banner.active{box-shadow:0 0 0 1px currentColor, 0 0 40px -8px currentColor}
  .banner.warn{color:var(--amber)} .banner.crit{color:var(--red)} .banner.pre{color:var(--amber)}
  .banner.crit.active{animation:pulse .9s ease-in-out infinite}
  .banner.pre.active{animation:pulse 1.4s ease-in-out infinite}
  @keyframes pulse{50%{box-shadow:0 0 0 1px currentColor,0 0 64px 0 currentColor}}
  /* LED 바 공통 */
  .bars{display:flex; flex-direction:column; gap:clamp(12px,2.2vw,24px); flex:1; justify-content:center}
  .bar{background:var(--panel); border:1px solid var(--dim); border-radius:14px;
       padding:clamp(12px,2vw,20px)}
  .bar .cap{display:flex; justify-content:space-between; align-items:baseline;
            margin-bottom:10px; font-size:clamp(13px,1.7vw,17px)}
  .bar .cap .val{font-variant-numeric:tabular-nums; font-weight:700; color:var(--txt)}
  .bar .cap .lbl{color:var(--sub); letter-spacing:.03em}
  .seg{position:relative; display:flex; gap:clamp(2px,.5vw,5px); height:clamp(30px,7vw,64px)}
  .seg i{flex:1; border-radius:4px; background:var(--dim); transition:background .09s linear,
         box-shadow .09s linear}
  /* 임계 마커 */
  .mark{position:absolute; top:-7px; bottom:-7px; width:2px; background:var(--sub); opacity:.55}
  .mark span{position:absolute; top:-16px; left:50%; transform:translateX(-50%);
             font-size:11px; color:var(--sub); white-space:nowrap}
  .prox-tiers{position:absolute; inset:0; pointer-events:none}
  .dir{font-size:clamp(20px,3.4vw,34px); font-weight:800; min-width:1.4em; text-align:center}
  .foot{text-align:center; color:var(--sub); font-size:12px}
</style></head><body>
  <div class="top">
    <div class="brand">AIRACLE <small>실시간 사이렌 감지 · LED</small></div>
    <div id="conn">연결 중…</div>
  </div>

  <div class="banner" id="banner">
    <div class="st" id="st">대기</div>
    <div class="meta">
      <div class="lv" id="lv"></div>
      <div class="row">
        <span id="subtype"></span><span id="risk"></span><span id="proxTxt"></span>
      </div>
    </div>
  </div>

  <div class="bars">
    <div class="bar">
      <div class="cap"><span class="lbl">감지 강도 (margin)</span>
        <span class="val" id="mVal">—</span></div>
      <div class="seg" id="mSeg"></div>
    </div>
    <div class="bar">
      <div class="cap"><span class="lbl">거리감 (배경 대비 ΔdB · 절대거리 아님)</span>
        <span class="val"><span class="dir" id="dir"></span><span id="pVal">—</span></span></div>
      <div class="seg" id="pSeg"></div>
    </div>
  </div>

  <div class="foot" id="foot"></div>

<script>
const N = 28;                          // LED 세그먼트 수
const mSeg = document.getElementById('mSeg'), pSeg = document.getElementById('pSeg');
for(const s of [mSeg,pSeg]) for(let i=0;i<N;i++) s.appendChild(document.createElement('i'));

// margin: [-2 .. +8] → [0..1].  prox_db: [0 .. 30] → [0..1].
const M_LO=-2, M_HI=8, P_LO=0, P_HI=30;
const clamp01=x=>Math.max(0,Math.min(1,x));
const mNorm = m => clamp01((m - M_LO)/(M_HI - M_LO));
const pNorm = p => clamp01((p - P_LO)/(P_HI - P_LO));

// 임계 마커(τ_on, τ_crit) — margin 바에 세로선. tier 경계(10,20dB) — prox 바.
function placeMark(seg, frac, label){
  const el=document.createElement('div'); el.className='mark';
  el.style.left=(frac*100)+'%';
  if(label){const s=document.createElement('span'); s.textContent=label; el.appendChild(s);}
  seg.appendChild(el);
}
placeMark(mSeg, mNorm(TAU_ON_PLACEHOLDER), 'τ');
placeMark(mSeg, mNorm(TAU_CRIT_PLACEHOLDER), '위험');
placeMark(pSeg, pNorm(10), '중간');
placeMark(pSeg, pNorm(20), '근접');

// 상태별 색/라벨.
const STLABEL={OFF:'대기',RISING:'감지 ↑',ON:'● 경보',FALLING:'유지 ↓',PRE:'◐ 예비',STALE:'신호 없음'};
const THEME={CRITICAL:'crit',WARN:'warn',PRE:'pre'};
const ACTIVE=new Set(['ON','FALLING','PRE']);
function colorFor(theme){ return theme==='crit'?getCSS('--red'):
  (theme==='warn'||theme==='pre')?getCSS('--amber'):getCSS('--green'); }
const getCSS=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// 목표값(수신) ← → 표시값(보간). tick 6.7Hz라도 60fps로 부드럽게.
let tgtM=0, tgtP=0, curM=0, curP=0, active=false, theme='', stale=true;
let litColor=getCSS('--green');

function render(){
  curM += (tgtM-curM)*0.18;  curP += (tgtP-curP)*0.18;   // ease
  paint(mSeg, curM, litColor);
  paint(pSeg, active?curP:0, litColor);
  requestAnimationFrame(render);
}
function paint(seg, frac, color){
  const lit=Math.round(frac*N), segs=seg.querySelectorAll('i');
  for(let i=0;i<N;i++){
    const on=i<lit;
    segs[i].style.background = on?color:getCSS('--dim');
    segs[i].style.boxShadow = on?('0 0 8px -1px '+color):'none';
  }
}
requestAnimationFrame(render);

const banner=document.getElementById('banner');
function apply(p){
  stale=p.stale; const t=p.tick||{};
  const conn=document.getElementById('conn');
  const hasData = t.state!==undefined && !stale;
  conn.textContent = stale?'런타임 대기':'● 수신 중';
  conn.className = stale?'':'on';

  // stale(런타임 무수신)면 '경보' 얼어붙지 않게 '신호 없음'으로 명시 전환
  const st = hasData ? (t.state||'OFF') : 'STALE';
  active = hasData && ACTIVE.has(st);
  theme = hasData ? (THEME[t.level]||'') : '';
  litColor = active ? colorFor(theme) : getCSS('--green');

  document.getElementById('st').textContent = STLABEL[st]||st;
  document.getElementById('st').style.color = active?litColor:(hasData?getCSS('--txt'):getCSS('--sub'));
  banner.className = 'banner'+(theme?(' '+theme):'')+(active?' active':'');

  document.getElementById('lv').textContent =
    active ? ({CRITICAL:'위급 · CRITICAL',WARN:'경보 · WARN',PRE:'예비경보 · PRE'}[t.level]||t.level||'') : '';
  document.getElementById('subtype').innerHTML = (hasData&&t.subtype)?('차종 <b>'+t.subtype+'</b>'):'';
  document.getElementById('risk').innerHTML = (active&&t.risk)?('위험도 <b>'+t.risk+'</b>'):'';
  document.getElementById('proxTxt').innerHTML = (active&&t.prox)?('거리감 <b>'+t.prox+'</b>'):'';

  // margin 바(항상 갱신 — 미검출시 ambient energy도 흔들림 보임)
  const m = (typeof t.margin==='number' && hasData)? t.margin : M_LO;
  tgtM = mNorm(m);
  document.getElementById('mVal').textContent = hasData? (m>=0?'+':'')+m.toFixed(1) : '—';

  // 거리감 바 + 방향 화살표
  const pdb = t.prox_db;
  const dirEl=document.getElementById('dir');
  if(active && typeof pdb==='number'){
    tgtP=pNorm(pdb);
    document.getElementById('pVal').textContent=(pdb>=0?'+':'')+pdb.toFixed(1)+' dB';
    dirEl.textContent = t.dir==='접근'?'▲':t.dir==='멀어짐'?'▼':t.dir==='정지'?'■':'';
    dirEl.style.color = t.dir==='접근'?getCSS('--red'):getCSS('--sub');
  } else {
    tgtP=0; document.getElementById('pVal').textContent='—'; dirEl.textContent='';
  }

  const ev=(p.events||[]).slice(-1)[0];
  document.getElementById('foot').textContent = ev
    ? ('최근 이벤트: '+({onset:'경보 발생',remind:'경보 지속',clear:'해제'}[ev.tag]||ev.tag)
       +'  ·  '+(ev.label||ev.kind||'')+'  t='+(ev.t==null?'':ev.t)+'s')
    : 'UDP 8737 수신 대기 — infer_trt --output console,udp';
}

// SSE 연결(자동 재접속).
function connect(){
  const es=new EventSource('/events');
  es.onmessage=e=>{ try{ apply(JSON.parse(e.data)); }catch(_){} };
  es.onerror=()=>{ es.close(); document.getElementById('conn').textContent='재연결…';
                   setTimeout(connect, 1000); };
}
connect();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                      # 접속 로그 소음 억제
        pass

    def do_GET(self):
        if self.path.startswith("/events"):
            return self._sse()
        body = PAGE.replace("TAU_ON_PLACEHOLDER", repr(TAU_ON)) \
                   .replace("TAU_CRIT_PLACEHOLDER", repr(TAU_CRIT)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_seq, last_beat = -1, 0.0
        try:
            while True:
                with _LOCK:
                    seq, stale = _SEQ, (time.monotonic() - _T_RECV > 1.5 if _T_RECV else True)
                now = time.monotonic()
                # tick 갱신 시 즉시 push, 그 외 0.4s 하트비트(stale 페이드·프록시 유지)
                if seq != last_seq or stale or (now - last_beat) > 0.4:
                    last_seq, last_beat = seq, now
                    msg = f"data: {_snapshot()}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return                                  # 브라우저 탭 닫힘 — 조용히 종료


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Airacle 실시간 LED바 디스플레이(UDP→SSE 웹)")
    ap.add_argument("--udp-host", default="127.0.0.1", help="UDP 수신 바인드(런타임과 동일 기기면 로컬)")
    ap.add_argument("--udp-port", type=int, default=8737, help="UdpSink 포트(기본 8737)")
    ap.add_argument("--http-host", default="0.0.0.0", help="웹 바인드(LAN 브라우저 접속 허용)")
    ap.add_argument("--http-port", type=int, default=8080)
    args = ap.parse_args(argv)

    threading.Thread(target=_udp_loop, args=(args.udp_host, args.udp_port), daemon=True).start()
    srv = ThreadingHTTPServer((args.http_host, args.http_port), Handler)
    srv.daemon_threads = True
    print(f"[led] 웹 http://{args.http_host}:{args.http_port}  (같은 기기: http://localhost:{args.http_port})",
          file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[led] 종료.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

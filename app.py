import json
import os

import streamlit as st

try:  # older Streamlit versions only
    import streamlit.components.v1 as components
except Exception:
    components = None

st.set_page_config(page_title="AgriBot - Weeding Robots", page_icon="🌱", layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:1.5rem}</style>",
            unsafe_allow_html=True)

SIM_HTML = """
<style>
 *{box-sizing:border-box;font-family:system-ui,-apple-system,Segoe UI,sans-serif}
 body{margin:0;color:#1b2b1b}
 .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
 canvas{width:100%;max-width:640px;height:auto;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.14)}
 .panel{flex:1;min-width:240px}
 .card{background:#fff;border-radius:12px;padding:10px 14px;margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
 .big{font-size:1.6rem;font-weight:700;color:#1b5e20;line-height:1.1}
 .lbl{font-size:.8rem;color:#5b6b5b}
 .dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px}
 .btns{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
 button,select{border:0;border-radius:9px;padding:8px 14px;font-size:.9rem;cursor:pointer;background:#2e7d32;color:#fff}
 button.sec{background:#e6f2e4;color:#1b5e20}
 select{background:#e6f2e4;color:#1b5e20}
 input[type=range]{width:100%;accent-color:#2e7d32}
 .legend span{display:inline-block;margin-right:10px;font-size:.78rem;color:#4b5b4b}
 #log{max-height:120px;overflow:auto;font-size:.85rem;line-height:1.5}
</style>
<div class="row">
  <canvas id="c" width="640" height="640"></canvas>
  <div class="panel">
    <div class="card"><div class="lbl">Step</div><div class="big"><span id="step">0</span> / <span id="total"></span></div></div>
    <div class="card"><div class="lbl">Weeds left</div><div class="big" id="left"></div>
      <div id="robots" style="margin-top:6px;font-size:.88rem"></div></div>
    <div class="card" style="border-left:5px solid #7b3fc4">
      <div class="lbl">Supervisor</div><div id="sup" style="font-weight:600;color:#7b3fc4"></div>
      <div id="fault" style="font-size:.8rem;color:#8a4b00;margin:4px 0"></div>
      <div id="log"></div></div>
    <div class="btns">
      <button id="play">Pause</button>
      <button id="restart" class="sec">Restart</button>
      <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option>
        <option value="2">2x</option><option value="4">4x</option></select>
    </div>
    <input type="range" id="slider" min="0" value="0" step="0.1">
    <div class="legend" style="margin-top:8px">
      <span style="color:#1e66e5">&#9632; Robot 1</span><span style="color:#f28c1b">&#9632; Robot 2</span>
      <span style="color:#7b3fc4">&#9679; Supervisor</span><span style="color:#2e7d32">&#9679; Crop</span>
      <span style="color:#d32f2f">&#9679; Weed</span>
    </div>
  </div>
</div>
<script>
var D = __DATA__;
var N = D.steps, half = D.field_size, S = 640, M = 30;
var scale = (S - 2 * M) / (2 * half);
var COLORS = ['#1e66e5', '#f28c1b'], SUPCOL = '#7b3fc4';
var BASE = 12;
var cv = document.getElementById('c'), ctx = cv.getContext('2d');
var slider = document.getElementById('slider'), playBtn = document.getElementById('play');
slider.max = N;
document.getElementById('total').textContent = N;
if (D.fault) {
  document.getElementById('fault').textContent = 'Test: ' + D.names[D.fault.robot] +
    ' steering is deliberately sabotaged (steps ' + D.fault.start + '-' + D.fault.end + ').';
}
var t = 0, playing = false, speed = 1, last = null, started = false, lastLog = '';

function X(x) { return M + (x + half) * scale; }
function Y(y) { return S - (M + (y + half) * scale); }

function drawField() {
  ctx.clearRect(0, 0, S, S);
  ctx.fillStyle = '#e9f3e4'; ctx.fillRect(0, 0, S, S);
  ctx.fillStyle = '#f4faf0'; ctx.fillRect(M, M, S - 2 * M, S - 2 * M);
  ctx.strokeStyle = '#d3e6cb'; ctx.lineWidth = 1;
  for (var g = -half; g <= half; g++) {
    ctx.beginPath(); ctx.moveTo(X(g), Y(-half)); ctx.lineTo(X(g), Y(half)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(-half), Y(g)); ctx.lineTo(X(half), Y(g)); ctx.stroke();
  }
  ctx.strokeStyle = '#8fbf86'; ctx.lineWidth = 2; ctx.strokeRect(M, M, S - 2 * M, S - 2 * M);
}

function circle(x, y, r, fill, stroke) {
  ctx.beginPath(); ctx.arc(X(x), Y(y), r, 0, 6.2832);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
}

function lerp(P, tt) {
  var k = Math.min(Math.floor(tt), N), k2 = Math.min(k + 1, N), f = tt - k;
  return [P[k][0] + (P[k2][0] - P[k][0]) * f, P[k][1] + (P[k2][1] - P[k][1]) * f];
}

function poseAt(i, tt) {
  var P = D.robots[i], k = Math.min(Math.floor(tt), N), k2 = Math.min(k + 1, N), f = tt - k;
  var a = P[k], b = P[k2];
  var dy = ((b[2] - a[2] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + dy * f];
}

function reasonAt(i, idx) {
  for (var j = 0; j < D.events.length; j++) {
    var e = D.events[j];
    if (e.robot === i && idx >= e.start && idx <= e.end) return e.reason;
  }
  return '';
}

function draw() {
  var idx = Math.min(Math.floor(t), N);
  drawField();
  var j, i;
  for (j = 0; j < D.crops.length; j++) circle(D.crops[j][0], D.crops[j][1], 11, '#66bb6a', '#2e7d32');
  var removedBy = [0, 0], left = 0;
  for (j = 0; j < D.weeds.length; j++) {
    var w = D.weeds[j];
    if (w.removed_step === null || idx < w.removed_step) {
      left++; circle(w.x, w.y, 9, '#d32f2f', '#7f1d1d');
    } else {
      removedBy[w.by]++;
      var age = t - w.removed_step;
      if (age >= 0 && age < 8) {
        ctx.globalAlpha = 1 - age / 8;
        circle(w.x, w.y, 9 + age * 3, null, COLORS[w.by % 2]);
        ctx.globalAlpha = 1;
      }
    }
  }
  for (i = 0; i < D.robots.length; i++) {
    ctx.strokeStyle = COLORS[i % 2]; ctx.globalAlpha = 0.3; ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (var k = 0; k <= idx; k++) {
      var q = D.robots[i][k];
      if (k === 0) ctx.moveTo(X(q[0]), Y(q[1])); else ctx.lineTo(X(q[0]), Y(q[1]));
    }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  var sp = lerp(D.supervisor, t);
  var msg = 'Supervisor: all robots working correctly', bcol = '#2e7d32', supMsg = 'All robots working correctly';
  for (i = 0; i < D.robots.length; i++) {
    if (D.guided[i][idx] === 1) {
      var rp = poseAt(i, t);
      ctx.setLineDash([7, 5]); ctx.strokeStyle = SUPCOL; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(X(sp[0]), Y(sp[1])); ctx.lineTo(X(rp[0]), Y(rp[1])); ctx.stroke();
      ctx.setLineDash([]);
      circle(rp[0], rp[1], 30, null, SUPCOL);
      if (bcol !== SUPCOL) {
        supMsg = 'Guiding ' + D.names[i] + ': ' + reasonAt(i, idx);
        msg = 'Supervisor is guiding ' + D.names[i] + ': ' + reasonAt(i, idx);
        bcol = SUPCOL;
      }
    }
  }
  for (i = 0; i < D.robots.length; i++) {
    var p = poseAt(i, t), c2 = COLORS[i % 2];
    ctx.setLineDash([4, 4]); ctx.globalAlpha = 0.45;
    circle(p[0], p[1], D.weed_reach * scale, null, c2);
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.save(); ctx.translate(X(p[0]), Y(p[1])); ctx.rotate(-p[2]);
    ctx.fillStyle = c2; ctx.fillRect(-20, -14, 40, 28);
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.moveTo(20, 0); ctx.lineTo(8, -8); ctx.lineTo(8, 8); ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  ctx.globalAlpha = 0.3; circle(sp[0], sp[1], 24, SUPCOL, null); ctx.globalAlpha = 1;
  circle(sp[0], sp[1], 14, SUPCOL, '#ffffff'); circle(sp[0], sp[1], 5, '#ffffff', null);
  ctx.fillStyle = SUPCOL; ctx.font = 'bold 12px sans-serif';
  ctx.fillText('Supervisor', X(sp[0]) - 28, Y(sp[1]) + 38);
  ctx.fillStyle = bcol; ctx.fillRect(M, 6, S - 2 * M, 24);
  ctx.fillStyle = '#ffffff'; ctx.font = 'bold 14px sans-serif'; ctx.fillText(msg, M + 10, 23);

  document.getElementById('step').textContent = idx;
  document.getElementById('left').textContent = left;
  document.getElementById('sup').textContent = (idx >= N && left === 0) ? 'Mission complete: field cleared' : supMsg;
  var h = '';
  for (i = 0; i < D.robots.length; i++) {
    h += '<div><span class="dot" style="background:' + COLORS[i % 2] + '"></span>' + D.names[i] + ': ' + removedBy[i] + ' removed</div>';
  }
  document.getElementById('robots').innerHTML = h;
  var log = '';
  for (j = 0; j < D.events.length; j++) {
    var e = D.events[j];
    if (e.start <= idx) log += '<div>Step ' + e.start + ': <b>' + e.robot_name + '</b> - ' + e.reason + (e.injected ? ' <i>(test fault)</i>' : '') + '</div>';
  }
  if (!log) log = '<div style="color:#5b6b5b">No corrections so far</div>';
  if (log !== lastLog) { document.getElementById('log').innerHTML = log; lastLog = log; }
}

function setBtn() { playBtn.textContent = playing ? 'Pause' : (t >= N ? 'Replay' : 'Play'); }

function loop(ts) {
  if (last === null) last = ts;
  var dt = (ts - last) / 1000; last = ts;
  if (playing) {
    t += dt * BASE * speed;
    if (t >= N) { t = N; playing = false; setBtn(); }
  }
  slider.value = t;
  draw();
  requestAnimationFrame(loop);
}

playBtn.onclick = function () { if (!playing && t >= N) t = 0; playing = !playing; setBtn(); };
document.getElementById('restart').onclick = function () { t = 0; playing = true; setBtn(); };
document.getElementById('speed').onchange = function (e) { speed = parseFloat(e.target.value); };
slider.oninput = function () { t = parseFloat(slider.value); };

function startOnce() { if (!started) { started = true; t = 0; playing = true; setBtn(); } }
if ('IntersectionObserver' in window) {
  new IntersectionObserver(function (es) { if (es[0].isIntersecting) startOnce(); }, {threshold: 0.3}).observe(cv);
} else { startOnce(); }
draw(); setBtn(); requestAnimationFrame(loop);
</script>
"""

# ---------------- load data ----------------
if not os.path.exists("stats.json"):
    st.error("stats.json not found. Run run_mission.py first.")
    st.stop()
with open("stats.json", encoding="utf-8") as f:
    s = json.load(f)
team, metrics = s["team"], s["metrics"]
sup = team.get("supervisor", {"interventions": 0, "events": [], "fault_injected": False})
traj = None
if os.path.exists("trajectory.json"):
    with open("trajectory.json", encoding="utf-8") as f:
        traj = json.load(f)

# ---------------- page ----------------
st.title("🌱 AgriBot: Weeding Robots with a Supervisor")
st.write("Two robots remove weeds from a field of crops. A third robot, the **Supervisor**, "
         "watches them and guides any robot that goes wrong.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Weeds removed", f"{team['weeds_removed']} / {team['total_weeds']}")
m2.metric("Steps", team["steps"])
m3.metric("Crop touches", team["crop_touches"])
m4.metric("Supervisor corrections", sup["interventions"])

st.subheader("Simulation")
if traj and "supervisor" in traj:
    sim_html = SIM_HTML.replace("__DATA__", json.dumps(traj))
    if hasattr(st, "iframe"):
        st.iframe(sim_html, height=700)
    else:
        components.html(sim_html, height=700, scrolling=True)
    st.caption("Replay of the recorded run. Use Play/Pause, the speed menu and the slider.")
else:
    st.info("Simulation data not found. Run run_mission.py to generate trajectory.json.")

st.subheader("Supervisor log")
if sup["events"]:
    for e in sup["events"]:
        tag = " (test fault)" if e.get("injected") else ""
        st.write(f"Step {e['start']}: **{e['robot_name']}**: {e['reason']}, Supervisor guided it back{tag}")
else:
    st.success("No corrections were needed. Both robots worked correctly.")
if sup.get("fault_injected"):
    st.info("This run included a deliberate fault test: one robot's steering was sabotaged "
            "to check that the Supervisor can detect and fix it.")

st.subheader("Farm report")
st.markdown(s["report"])
st.caption(f"Written by: {s['report_source']}. Business figures are estimates based on "
           f"assumptions: {metrics['assumptions']}")

st.subheader("Email alerts")
for a in s["alerts"]:
    text = f"**{a['type']}**: {a['subject']}  ({a['detail']})"
    (st.success if a["sent"] else st.warning)(text)

with st.expander("PyBullet video"):
    if os.path.exists("demo.mp4"):
        st.video("demo.mp4")
    else:
        st.info("demo.mp4 not found.")
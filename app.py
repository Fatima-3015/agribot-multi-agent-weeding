import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgriBot - Multi-Agent Weeding Robots",
                   page_icon="🌱", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#1b5e20 0%,#43a047 60%,#7cb342 100%);
       padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {color:#ffffff !important; margin:0 0 .3rem 0; font-size:2.1rem;}
.hero p {color:#ffffff !important; margin:0; opacity:.95; font-size:1.05rem;}
.badge {display:inline-block; background:rgba(255,255,255,.22); color:#fff;
        padding:.2rem .75rem; border-radius:999px; font-size:.8rem; margin:.8rem .4rem 0 0;}
.kpi {background:#ffffff; border-radius:14px; padding:1rem 1.2rem;
      box-shadow:0 2px 10px rgba(0,0,0,.07); border-left:6px solid #43a047;}
.kpi.warn {border-left-color:#e53935;}
.kpi .v {font-size:1.9rem; font-weight:700; color:#1b5e20; line-height:1.15;}
.kpi.warn .v {color:#c62828;}
.kpi .l {font-size:.85rem; color:#5b6b5b;}
.sec {font-size:1.25rem; font-weight:700; color:#1b5e20; margin:1.4rem 0 .6rem 0;}
.robot {background:#ffffff; border-radius:12px; padding:.8rem 1rem; margin-bottom:.7rem;
        box-shadow:0 2px 8px rgba(0,0,0,.06);}
.dot {display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:.5rem;}
.robot .n {font-weight:600;}
.robot .s {color:#5b6b5b; font-size:.9rem; margin-top:.2rem;}
.agent {background:#ffffff; border-radius:14px; padding:1rem; height:100%;
        box-shadow:0 2px 10px rgba(0,0,0,.07); text-align:center;}
.agent .i {font-size:1.9rem;}
.agent .t {font-weight:700; color:#1b5e20; margin:.3rem 0 .2rem 0; font-size:.95rem;}
.agent .d {color:#5b6b5b; font-size:.82rem;}
.agent .step {color:#7cb342; font-weight:700; font-size:.75rem;}
.foot {color:#6b7b6b; font-size:.82rem; margin-top:2rem; text-align:center;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- load data ----------------
if not os.path.exists("stats.json"):
    st.error("stats.json not found. Run run_mission.py first.")
    st.stop()
with open("stats.json", encoding="utf-8") as f:
    s = json.load(f)
team, metrics = s["team"], s["metrics"]
baseline = s.get("baseline")

# ---------------- hero ----------------
st.markdown(
    '<div class="hero"><h1>🌱 AgriBot: Multi-Agent Weeding Robots</h1>'
    '<p>Two PPO-trained robots, coordinated by a Planner agent, clear weeds without '
    'touching the crops. Gemini writes the farm report and email alerts go out automatically.</p>'
    '<span class="badge">PyBullet simulation</span><span class="badge">PPO reinforcement learning</span>'
    '<span class="badge">Multi-agent</span><span class="badge">Gemini AI report</span>'
    '<span class="badge">Email automation</span></div>',
    unsafe_allow_html=True)


# ---------------- KPI cards ----------------
def kpi(value, label, warn=False):
    cls = "kpi warn" if warn else "kpi"
    return f'<div class="{cls}"><div class="v">{value}</div><div class="l">{label}</div></div>'


saved = metrics.get("time_saved_vs_single_robot_pct")
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi(f"{team['weeds_removed']} / {team['total_weeds']}", "Weeds removed"), unsafe_allow_html=True)
k2.markdown(kpi(team["steps"], "Steps to clear the field"), unsafe_allow_html=True)
k3.markdown(kpi(team["crop_touches"], "Crop touches", warn=team["crop_touches"] > 0), unsafe_allow_html=True)
k4.markdown(kpi(f"{saved}%" if saved is not None else "n/a", "Faster than 1 robot"), unsafe_allow_html=True)

# ---------------- video + robots ----------------
st.markdown('<div class="sec">🎥 Mission replay</div>', unsafe_allow_html=True)
left, right = st.columns([3, 2])
with left:
    if os.path.exists("demo.mp4"):
        st.video("demo.mp4")
    else:
        st.error("demo.mp4 not found.")
    st.caption(f"Controller: {s['controller']}  |  Generated: {s['generated_at']}")

with right:
    colors = ["#1e66e5", "#f28c1b"]
    for i, r in enumerate(team["per_robot"]):
        st.markdown(
            f'<div class="robot"><div class="n"><span class="dot" style="background:{colors[i % 2]}"></span>'
            f'{r["robot"]}</div><div class="s">🌿 {r["weeds_removed"]} weeds removed &nbsp;|&nbsp; '
            f'⚠️ {r["crop_touches"]} crop touches</div></div>',
            unsafe_allow_html=True)
    if baseline:
        st.markdown("**1 robot vs 2 robots**")
        df = pd.DataFrame({"Steps": [baseline["steps"], team["steps"]]},
                          index=["1 robot", "2 robots"])
        st.bar_chart(df, color="#43a047", height=220)

# ---------------- agents ----------------
st.markdown('<div class="sec">🧠 How the agents work together</div>', unsafe_allow_html=True)
agents = [
    ("👁️", "Perception Agent", "Senses weeds, crops and robot positions"),
    ("🧭", "Planner Agent", "Gives each robot its own weed, no overlap"),
    ("🤖", "Controller Agents x2", "One trained PPO brain drives each robot"),
    ("📝", "Report Agent (Gemini)", "Writes the farm report from mission stats"),
    ("📧", "Alert Agent", "Emails mission report, crop damage and incomplete alerts"),
]
for col, (icon, title, desc), n in zip(st.columns(5), agents, range(1, 6)):
    col.markdown(
        f'<div class="agent"><div class="step">STEP {n}</div><div class="i">{icon}</div>'
        f'<div class="t">{title}</div><div class="d">{desc}</div></div>',
        unsafe_allow_html=True)

# ---------------- tabs ----------------
st.markdown('<div class="sec">📊 Results</div>', unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["📝 Farm report", "💼 Business impact", "📧 Alerts", "ℹ️ About & limits"])

with t1:
    st.markdown(s["report"])
    st.caption(f"Written by: {s['report_source']}")

with t2:
    b1, b2, b3 = st.columns(3)
    b1.metric("Manual labour equivalent", f"{metrics['manual_labour_minutes_equivalent']} min")
    b2.metric("Labour cost equivalent", f"PKR {metrics['manual_labour_cost_pkr_equivalent']}")
    b3.metric("Estimated crop loss", f"PKR {metrics['estimated_crop_loss_pkr']}")
    st.caption("These are estimates based on assumptions, not measured data: "
               f"{metrics['assumptions']}")

with t3:
    for a in s["alerts"]:
        text = f"**{a['type']}**: {a['subject']}  ({a['detail']})"
        (st.success if a["sent"] else st.warning)(text)

with t4:
    st.markdown(
        "- This is a **simulation** (PyBullet), not a real robot.\n"
        "- The field layout is fixed and the PPO model was trained on it.\n"
        f"- Two robots finished in {team['steps']} steps"
        + (f" vs {baseline['steps']} for one robot." if baseline else ".") + "\n"
        "- Business figures are assumption-based estimates.")

st.markdown('<div class="foot">AgriBot · Multi-Agent + AI Business Automation · Built with PPO, PyBullet, Gemini and Streamlit</div>',
            unsafe_allow_html=True)
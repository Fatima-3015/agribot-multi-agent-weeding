import json
import os

import streamlit as st

st.set_page_config(page_title="AgriBot - Multi-Agent Weeding Robots",
                   page_icon="🌱", layout="centered")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {background: linear-gradient(120deg,#1b5e20 0%,#43a047 100%);
       padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem; text-align:center;}
.hero h1 {color:#ffffff !important; margin:0 0 .3rem 0; font-size:1.9rem;}
.hero p {color:#eafaea !important; margin:0; font-size:1rem;}
.kpi {background:#ffffff; border-radius:14px; padding:1rem; text-align:center;
      box-shadow:0 2px 10px rgba(0,0,0,.07); border-left:6px solid #43a047;}
.kpi .v {font-size:1.7rem; font-weight:700; color:#1b5e20;}
.kpi .l {font-size:.85rem; color:#5b6b5b;}
.note {background:#eef7ee; border-radius:12px; padding:1rem 1.2rem; color:#1b5e20;
       font-size:.95rem; margin-top:1rem;}
.foot {color:#6b7b6b; font-size:.8rem; margin-top:1.5rem; text-align:center;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🌱 AgriBot</h1>'
    '<p>Two robots clear weeds from a field without touching the crops.</p></div>',
    unsafe_allow_html=True)

# ---------------- load mission stats (from record_mission.py) ----------------
if not os.path.exists("stats.json"):
    st.error("stats.json not found. Run record_mission.py first and copy stats.json + demo.mp4 here.")
    st.stop()
with open("stats.json", encoding="utf-8") as f:
    s = json.load(f)
team = s["team"]

c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="kpi"><div class="v">{team["weeds_removed"]}/{team["total_weeds"]}</div>'
            f'<div class="l">Weeds removed</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><div class="v">{team["steps"]}</div>'
            f'<div class="l">Steps taken</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><div class="v">{team["crop_touches"]}</div>'
            f'<div class="l">Crop touches</div></div>', unsafe_allow_html=True)

# ---------------- video ----------------
st.subheader("🎥 Mission video")
if os.path.exists("demo.mp4"):
    st.video("demo.mp4")
else:
    st.warning("demo.mp4 not found. Run record_mission.py and copy demo.mp4 next to app.py.")

for r in team["per_robot"]:
    st.write(f"**{r['robot']}** \u2014 {r['weeds_removed']} weeds removed, {r['crop_touches']} crop touches")

# ---------------- supervisor note (keeps the "who checks the robots" idea, simply) ----------------
st.markdown(
    '<div class="note">🧭 <b>Planner Agent:</b> splits the field so each robot only works its own half, '
    'so they never both go for the same weed. If a robot touches a crop, that counts as a mistake and '
    'triggers an alert email below \u2014 that is how the system "catches" wrong behaviour.</div>',
    unsafe_allow_html=True)

# ---------------- alerts (email automation) ----------------
st.subheader("📧 Alerts sent")
for a in s.get("alerts", []):
    text = f"**{a['type']}**: {a['subject']}  ({a['detail']})"
    (st.success if a.get("sent") else st.warning)(text)

st.markdown('<div class="foot">AgriBot · Multi-Agent Weeding · PyBullet + PPO + Email Automation</div>',
            unsafe_allow_html=True)
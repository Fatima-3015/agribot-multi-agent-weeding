import json
import os

import streamlit as st

st.set_page_config(page_title="Autonomous Weeding Robots", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent Autonomous Weeding Robots")
st.write("Two PPO-trained robots, coordinated by a Planner agent, remove weeds from a "
         "fixed field of crops (PyBullet simulation, recorded run). A GenAI agent writes "
         "the farm report and email alerts are sent automatically.")

st.markdown("**Pipeline:** Perception Agent → Planner Agent → 2 Controller Agents (PPO) "
            "→ Mission Stats → Gemini Report → Email Alerts")

if not os.path.exists("stats.json"):
    st.error("stats.json not found. Run run_mission.py first.")
    st.stop()

with open("stats.json", encoding="utf-8") as f:
    s = json.load(f)
team, metrics = s["team"], s["metrics"]

left, right = st.columns([3, 2])

with left:
    if os.path.exists("demo.mp4"):
        st.video("demo.mp4")
    else:
        st.error("demo.mp4 not found.")
    st.caption(f"Controller: {s['controller']}  |  Generated: {s['generated_at']}")

with right:
    c1, c2 = st.columns(2)
    c1.metric("Weeds removed", f"{team['weeds_removed']} / {team['total_weeds']}")
    c2.metric("Steps", team["steps"])
    c3, c4 = st.columns(2)
    c3.metric("Crop touches", team["crop_touches"])
    saved = metrics.get("time_saved_vs_single_robot_pct")
    c4.metric("Faster than 1 robot", f"{saved}%" if saved is not None else "n/a")
    st.subheader("Per robot")
    st.table(team["per_robot"])

st.divider()
st.subheader("📝 AI Farm Report")
st.markdown(s["report"])
st.caption(f"Written by: {s['report_source']}. Business figures are estimates based on "
           f"assumptions: {metrics['assumptions']}")

st.subheader("📧 Alerts")
for a in s["alerts"]:
    icon = "✅" if a["sent"] else "⚠️"
    st.write(f"{icon} **{a['type']}** — {a['subject']}  _({a['detail']})_")

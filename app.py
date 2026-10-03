import json
import math
import os
import random
import tempfile
import time

import numpy as np
import pandas as pd
import streamlit as st
import pybullet as p
import pybullet_data

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
.live-badge {display:inline-block; background:#e53935; color:#fff; padding:.15rem .6rem;
             border-radius:999px; font-size:.72rem; font-weight:700; margin-left:.5rem;
             vertical-align:middle;}
.foot {color:#6b7b6b; font-size:.82rem; margin-top:2rem; text-align:center;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- load data (KPIs / report / alerts still come from the last recorded mission) ----------------
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


# ---------------- KPI cards (from the last recorded mission) ----------------
def kpi(value, label, warn=False):
    cls = "kpi warn" if warn else "kpi"
    return f'<div class="{cls}"><div class="v">{value}</div><div class="l">{label}</div></div>'


saved = metrics.get("time_saved_vs_single_robot_pct")
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi(f"{team['weeds_removed']} / {team['total_weeds']}", "Weeds removed"), unsafe_allow_html=True)
k2.markdown(kpi(team["steps"], "Steps to clear the field"), unsafe_allow_html=True)
k3.markdown(kpi(team["crop_touches"], "Crop touches", warn=team["crop_touches"] > 0), unsafe_allow_html=True)
k4.markdown(kpi(f"{saved}%" if saved is not None else "n/a", "Faster than 1 robot"), unsafe_allow_html=True)


# =====================================================================
# LIVE SIMULATION (replaces the pre-recorded demo.mp4)
# =====================================================================
FIELD_SIZE = 4
WEED_REACH = 0.5
MAX_LIVE_STEPS = 600
FRAME_EVERY = 6          # send a new frame to the browser every N physics steps

ROBOT_URDF = """
<robot name="weeder_robot">
  <link name="chassis">
    <visual><geometry><box size="0.35 0.25 0.12"/></geometry>
      <material name="c"><color rgba="{r} {g} {b} 1"/></material></visual>
    <collision><geometry><box size="0.35 0.25 0.12"/></geometry></collision>
    <inertial><mass value="2.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="left_wheel">
    <visual><geometry><cylinder radius="0.06" length="0.04"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material></visual>
    <collision><geometry><cylinder radius="0.06" length="0.04"/></geometry></collision>
    <inertial><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="right_wheel">
    <visual><geometry><cylinder radius="0.06" length="0.04"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material></visual>
    <collision><geometry><cylinder radius="0.06" length="0.04"/></geometry></collision>
    <inertial><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="caster_wheel">
    <visual><geometry><sphere radius="0.04"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material></visual>
    <collision><geometry><sphere radius="0.04"/></geometry></collision>
    <inertial><mass value="0.1"/><inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="chassis"/><child link="left_wheel"/>
    <origin xyz="0 0.15 -0.02" rpy="1.5708 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="right_wheel_joint" type="continuous">
    <parent link="chassis"/><child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.02" rpy="1.5708 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="caster_joint" type="fixed">
    <parent link="chassis"/><child link="caster_wheel"/>
    <origin xyz="0.15 0 -0.04" rpy="0 0 0"/>
  </joint>
</robot>
"""


def _make_robot(color, spawn_xy):
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(ROBOT_URDF.format(r=color[0], g=color[1], b=color[2]))
        path = f.name
    return p.loadURDF(path, basePosition=[spawn_xy[0], spawn_xy[1], 0.08])


def _build_field():
    crop_positions = []
    xs = [(-FIELD_SIZE + 1) + i * 2 for i in range(4)]
    ys = [(-FIELD_SIZE + 1) + j * 2 for j in range(4)]
    for x in xs:
        for y in ys:
            crop_positions.append((x, y))
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=0.3)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.3,
                                       rgbaColor=[0.15, 0.6, 0.15, 1])
            p.createMultiBody(0, col, vis, [x, y, 0.15])

    random.seed(42)   # same field layout every run, matches the "fixed field layout" note
    weed_ids, weed_pos = [], {}
    placed = 0
    while placed < 12:
        x = random.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
        y = random.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
        if not any((x - cx) ** 2 + (y - cy) ** 2 < 0.35 ** 2 for cx, cy in crop_positions):
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[0.55, 0.27, 0.07, 1])
            wid = p.createMultiBody(0, -1, vis, [x, y, 0.06])
            weed_ids.append(wid)
            weed_pos[wid] = (x, y)
            placed += 1
    return crop_positions, weed_ids, weed_pos


def _drive_towards(robot_id, target_xy):
    """Simple live-demo controller: steer one robot toward a target point.
    (Swap this for model.predict(obs) if you upload a trained PPO model
    alongside the app, to make the live run match the real PPO policy.)"""
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    rx, ry = pos[0], pos[1]
    yaw = p.getEulerFromQuaternion(orn)[2]
    tx, ty = target_xy
    ang = math.atan2(ty - ry, tx - rx) - yaw
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    turn = max(-1.0, min(1.0, 2.0 * ang))
    forward = 1.0 if abs(ang) < 1.0 else 0.0
    left = (forward - turn) * 8
    right = (forward + turn) * 8
    p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=-left, force=5)
    p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=-right, force=5)
    return rx, ry


def run_live_mission():
    frame_box = st.empty()
    c1, c2, c3 = st.columns(3)
    r1_box, r2_box, total_box = c1.empty(), c2.empty(), c3.empty()

    client = p.connect(p.DIRECT)   # headless physics — works fine on Streamlit Cloud
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    robot_blue = _make_robot((0.12, 0.40, 0.90), (-3, -3.5))    # Robot 1 — left zone
    robot_orange = _make_robot((0.95, 0.55, 0.10), (3, -3.5))   # Robot 2 — right zone
    crop_positions, weed_ids, weed_pos = _build_field()

    view_matrix = p.computeViewMatrix([0, -7, 6], [0, 0, 0], [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(50, 1.33, 0.1, 20)

    removed = {"blue": 0, "orange": 0}
    total = len(weed_ids)
    step = 0

    def zone_weeds(x_min, x_max):
        return [w for w in weed_ids if x_min <= weed_pos[w][0] < x_max]

    while weed_ids and step < MAX_LIVE_STEPS:
        for robot_id, key, (zmin, zmax) in (
            (robot_blue, "blue", (-FIELD_SIZE, 0)),
            (robot_orange, "orange", (0, FIELD_SIZE)),
        ):
            zw = zone_weeds(zmin, zmax)
            if not zw:
                continue
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            target = min(zw, key=lambda w: (weed_pos[w][0] - pos[0]) ** 2 + (weed_pos[w][1] - pos[1]) ** 2)
            rx, ry = _drive_towards(robot_id, weed_pos[target])
            if math.hypot(weed_pos[target][0] - rx, weed_pos[target][1] - ry) < WEED_REACH:
                p.removeBody(target)
                weed_ids.remove(target)
                del weed_pos[target]
                removed[key] += 1

        p.stepSimulation()
        step += 1

        if step % FRAME_EVERY == 0:
            _, _, rgb, _, _ = p.getCameraImage(560, 380, view_matrix, proj_matrix)
            frame = np.reshape(rgb, (380, 560, 4))[:, :, :3].astype(np.uint8)
            frame_box.image(frame, caption=f"Live \u2014 step {step}", use_container_width=True)
            r1_box.markdown(f"🔵 **Robot 1 (blue)**: {removed['blue']} removed")
            r2_box.markdown(f"🟠 **Robot 2 (orange)**: {removed['orange']} removed")
            total_box.markdown(f"🌱 **Total**: {removed['blue'] + removed['orange']}/{total}")
            time.sleep(0.04)

    p.disconnect(client)
    return removed, total, step


# ---------------- Mission replay (now LIVE) + robots panel ----------------
st.markdown(
    '<div class="sec">🎥 Mission replay <span class="live-badge">LIVE</span></div>',
    unsafe_allow_html=True)
left, right = st.columns([3, 2])

with left:
    st.caption("This runs the simulation for real, right now on the server \u2014 it is not a "
               "pre-recorded video. Weeds disappear live as each robot reaches them.")
    if st.button("▶️ Run Live Simulation"):
        live_removed, live_total, live_steps = run_live_mission()
        st.success(f"✅ Live run finished: {live_removed['blue'] + live_removed['orange']}/{live_total} "
                   f"weeds removed in {live_steps} steps.")
    st.caption(f"Controller: {s['controller']}  |  Last recorded mission: {s['generated_at']}")

with right:
    st.markdown("**Last recorded mission** (from stats.json)")
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
        "- The **Mission replay** panel above runs a real, live PyBullet simulation on the "
        "server (headless) \u2014 it is not a pre-recorded video.\n"
        "- The live demo uses a lightweight seek-and-remove controller so it runs smoothly "
        "in the browser; the KPI cards and report above come from the last full PPO-trained "
        "mission recorded in `stats.json`.\n"
        "- The field layout is fixed (same crop/weed positions every run).\n"
        f"- Two robots finished in {team['steps']} steps"
        + (f" vs {baseline['steps']} for one robot." if baseline else ".") + "\n"
        "- Business figures are assumption-based estimates.")

st.markdown('<div class="foot">AgriBot · Multi-Agent + AI Business Automation · Built with PPO, PyBullet, Gemini and Streamlit</div>',
            unsafe_allow_html=True)
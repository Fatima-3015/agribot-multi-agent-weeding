"""
Live Simulation Tab (drop-in piece for app.py)
-------------------------------------------------
This replaces a pre-recorded "Mission replay" video with a REAL, running
simulation: when the user clicks the button, the server runs the actual
PyBullet simulation and streams a new frame to the browser every few
steps, so weeds visibly disappear in real time as the robot reaches them.

How to use:
    1. Place this file next to app.py
    2. In app.py, add near the top:
         from live_simulation_tab import render_live_simulation_tab
    3. Inside whichever tab should show the live demo, call:
         render_live_simulation_tab()

Install (if not already):
    pip install pybullet numpy streamlit
"""

import random
import tempfile
import time

import numpy as np
import pybullet as p
import pybullet_data
import streamlit as st

FIELD_SIZE = 4
WEED_REACH = 0.5
MAX_STEPS = 600
FRAME_EVERY = 6          # capture + send a frame every N physics steps

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


def _make_robot(color):
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(ROBOT_URDF.format(r=color[0], g=color[1], b=color[2]))
        path = f.name
    return p.loadURDF(path, basePosition=[-3, -3.5, 0.08])


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

    random.seed(42)
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


def render_live_simulation_tab():
    st.subheader("\U0001F534 Live Simulation (runs for real, right now)")
    st.caption("This is not a pre-recorded video \u2014 the simulation runs on the server "
               "as you watch, and weeds disappear live as the robot reaches them.")

    if st.button("\u25B6\uFE0F Run Live Simulation"):
        frame_box = st.empty()
        status_box = st.empty()

        client = p.connect(p.DIRECT)   # headless physics, works on Streamlit Cloud
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        robot_id = _make_robot((0.2, 0.3, 0.9))
        crop_positions, weed_ids, weed_pos = _build_field()

        view_matrix = p.computeViewMatrix([0, -7, 6], [0, 0, 0], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(50, 1.33, 0.1, 20)

        removed = 0
        total = len(weed_ids)

        for step in range(MAX_STEPS):
            # simple "seek nearest weed" controller for a believable live demo
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            rx, ry = pos[0], pos[1]
            if weed_ids:
                tx, ty = min(
                    (weed_pos[w] for w in weed_ids),
                    key=lambda t: (t[0] - rx) ** 2 + (t[1] - ry) ** 2,
                )
                import math
                yaw = p.getEulerFromQuaternion(orn)[2]
                ang = math.atan2(ty - ry, tx - rx) - yaw
                ang = (ang + math.pi) % (2 * math.pi) - math.pi
                turn = max(-1.0, min(1.0, 2.0 * ang))
                forward = 1.0 if abs(ang) < 1.0 else 0.0
                left = (forward - turn) * 8
                right = (forward + turn) * 8
                p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=-left, force=5)
                p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=-right, force=5)

                if math.hypot(tx - rx, ty - ry) < WEED_REACH:
                    wid_to_remove = next(w for w in weed_ids if weed_pos[w] == (tx, ty))
                    p.removeBody(wid_to_remove)
                    weed_ids.remove(wid_to_remove)
                    del weed_pos[wid_to_remove]
                    removed += 1

            p.stepSimulation()

            if step % FRAME_EVERY == 0:
                _, _, rgb, _, _ = p.getCameraImage(480, 360, view_matrix, proj_matrix)
                frame = np.reshape(rgb, (360, 480, 4))[:, :, :3].astype(np.uint8)
                frame_box.image(frame, caption=f"Step {step}", use_container_width=True)
                status_box.info(f"\U0001F331 Weeds removed: {removed}/{total}")
                time.sleep(0.05)   # small pause so it visibly animates

            if not weed_ids:
                break

        status_box.success(f"\u2705 Done! {removed}/{total} weeds removed in {step} steps.")
        p.disconnect(client)

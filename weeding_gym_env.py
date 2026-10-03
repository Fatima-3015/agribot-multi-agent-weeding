"""
Gymnasium wrapper for the Autonomous Weeding Robot
------------------------------------------------------
This turns the rule-based robot/field (weeding_robot_env.py) into a
proper Reinforcement Learning environment. Instead of the fixed
steer_to() logic deciding how the robot moves, a PPO agent will learn
its own policy by trial and reward.

Observation (what the robot "sees" each step), all in the robot's own frame:
    [ dx_to_nearest_weed, dy_to_nearest_weed,
      dx_to_nearest_crop, dy_to_nearest_crop,
      weeds_remaining_fraction ]

Action (what the robot can do each step):
    [ turn, forward ]   both in range [-1, 1]  (continuous control)

Reward (matches the values shown in the project presentation):
    +10   weed removed
    -15   crop touched
    -0.1  every step (encourages efficiency)
    +shaping for moving closer to the nearest weed

Install:
    pip install gymnasium stable-baselines3 pybullet numpy
"""

import math
import random
import tempfile

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

# ---------------------------------------------------------------------
# Same settings used in the presentation / rule-based version
# ---------------------------------------------------------------------
FIELD_SIZE = 4
ROWS, COLS = 4, 4
NUM_WEEDS = 12
RANDOM_SEED = 42

WEED_REACH = 0.5          # distance threshold from the parameters slide
CROP_HIT_DIST = 0.18      # how close counts as "touching" a crop
MAX_SPEED = 10
MAX_STEPS = 500           # matches "maximum steps per episode" from the slide

# PyBullet runs at 240 Hz. One RL step = FRAME_SKIP physics steps (0.1 s).
# Without this the robot moves only ~0.0025 m per RL step and can never
# reach a weed within 500 steps.
FRAME_SKIP = 24

REWARD_WEED = 10
REWARD_CROP_HIT = -15
REWARD_STEP = -0.1
SHAPING_SCALE = 5.0       # reward per metre moved closer to the nearest weed

ROBOT_URDF = """
<robot name="weeder_robot">
  <link name="chassis">
    <visual><geometry><box size="0.35 0.25 0.12"/></geometry>
      <material name="grey"><color rgba="0.3 0.3 0.3 1"/></material></visual>
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


class WeedingEnv(gym.Env):
    """A Gymnasium environment: one episode = one attempt to clear the field."""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode="none"):
        super().__init__()
        self.render_mode = render_mode
        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)

        # action: [turn, forward], both scaled to [-1, 1] by the agent
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # observation: see the docstring at the top of the file
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(5,), dtype=np.float32)

        self.robot_id = None
        self.crop_positions = []
        self.weed_ids = []
        self.weed_pos = {}
        self.step_count = 0
        self.prev_nearest_weed_dist = None

    # -------------------------------------------------------------
    def _build_robot(self):
        with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
            f.write(ROBOT_URDF)
            path = f.name
        self.robot_id = p.loadURDF(path, basePosition=[0, 0, 0.08])

    def _build_field(self):
        self.crop_positions = []
        xs = [(-FIELD_SIZE + 1) + i * 2 for i in range(COLS)]
        ys = [(-FIELD_SIZE + 1) + j * 2 for j in range(ROWS)]
        for x in xs:
            for y in ys:
                self.crop_positions.append((x, y))
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=0.3)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.3,
                                           rgbaColor=[0.15, 0.6, 0.15, 1])
                p.createMultiBody(0, col, vis, [x, y, 0.15])

        random.seed(RANDOM_SEED)
        self.weed_ids, self.weed_pos = [], {}
        placed = 0
        while placed < NUM_WEEDS:
            x = random.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
            y = random.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
            too_close = any((x - cx) ** 2 + (y - cy) ** 2 < 0.35 ** 2 for cx, cy in self.crop_positions)
            if not too_close:
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[0.55, 0.27, 0.07, 1])
                wid = p.createMultiBody(0, -1, vis, [x, y, 0.06])
                self.weed_ids.append(wid)
                self.weed_pos[wid] = (x, y)
                placed += 1

    def _robot_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return pos[0], pos[1], yaw

    def _nearest(self, rx, ry, positions):
        if not positions:
            return 0.0, 0.0
        best, best_d = None, 1e9
        for px, py in positions:
            d = math.hypot(px - rx, py - ry)
            if d < best_d:
                best, best_d = (px, py), d
        return best[0] - rx, best[1] - ry

    def _get_obs(self):
        rx, ry, yaw = self._robot_pose()
        wdx, wdy = self._nearest(rx, ry, [self.weed_pos[w] for w in self.weed_ids])
        cdx, cdy = self._nearest(rx, ry, self.crop_positions)

        # rotate weed AND crop offsets into the robot's own facing direction
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        wdx_r = wdx * cos_y - wdy * sin_y
        wdy_r = wdx * sin_y + wdy * cos_y
        cdx_r = cdx * cos_y - cdy * sin_y
        cdy_r = cdx * sin_y + cdy * cos_y

        frac_left = len(self.weed_ids) / NUM_WEEDS
        return np.array([wdx_r, wdy_r, cdx_r, cdy_r, frac_left], dtype=np.float32)

    # -------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self._build_robot()
        self._build_field()
        self.step_count = 0
        for _ in range(30):
            p.stepSimulation()

        rx, ry, _ = self._robot_pose()
        self.prev_nearest_weed_dist = self._nearest_weed_distance(rx, ry)
        return self._get_obs(), {}

    def _nearest_weed_distance(self, rx, ry):
        if not self.weed_ids:
            return 0.0
        return min(math.hypot(self.weed_pos[w][0] - rx, self.weed_pos[w][1] - ry)
                   for w in self.weed_ids)

    def step(self, action):
        turn, forward = float(action[0]), float(action[1])
        left = (forward - turn) * MAX_SPEED
        right = (forward + turn) * MAX_SPEED
        p.setJointMotorControl2(self.robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=-left, force=5)
        p.setJointMotorControl2(self.robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=-right, force=5)
        for _ in range(FRAME_SKIP):       # let the robot actually move
            p.stepSimulation()
        self.step_count += 1

        rx, ry, _ = self._robot_pose()
        reward = REWARD_STEP
        terminated = False

        # reward shaping: bonus for getting closer to the nearest weed,
        # penalty for moving away from it. Without this, reward is too
        # sparse and the agent never gets a learning signal.
        nearest_dist = self._nearest_weed_distance(rx, ry)
        if self.prev_nearest_weed_dist is not None:
            reward += SHAPING_SCALE * (self.prev_nearest_weed_dist - nearest_dist)
        self.prev_nearest_weed_dist = nearest_dist

        # did we reach a weed?
        for wid in list(self.weed_ids):
            wx, wy = self.weed_pos[wid]
            if math.hypot(wx - rx, wy - ry) < WEED_REACH:
                p.removeBody(wid)
                self.weed_ids.remove(wid)
                del self.weed_pos[wid]
                reward += REWARD_WEED
                self.prev_nearest_weed_dist = self._nearest_weed_distance(rx, ry)

        # did we touch a crop?
        for cx, cy in self.crop_positions:
            if math.hypot(cx - rx, cy - ry) < CROP_HIT_DIST:
                reward += REWARD_CROP_HIT

        if not self.weed_ids:
            terminated = True  # field fully cleared

        truncated = self.step_count >= MAX_STEPS
        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        if p.isConnected(self._client):
            p.disconnect(self._client)
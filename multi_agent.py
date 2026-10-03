"""
Multi-Agent Weeding System
------------------------------------------------------
Agents:
    Perception Agent  -> senses weeds, crops and robot poses in the field
    Planner Agent     -> assigns a different weed to every robot (no overlap)
    Controller Agents -> one per robot; each uses the SAME trained PPO model
                         (falls back to a rule-based steering if no model)

The Planner's assigned weed is what each robot "sees" as its nearest weed, so
the already-trained PPO model works without any re-training.
"""

import math
import os
import random
import tempfile

import numpy as np
import pybullet as p
import pybullet_data

from weeding_gym_env import (
    FIELD_SIZE, ROWS, COLS, NUM_WEEDS, RANDOM_SEED, WEED_REACH,
    CROP_HIT_DIST, MAX_SPEED, MAX_STEPS, FRAME_SKIP, ROBOT_URDF,
)

RECORD_EVERY = 2                      # save every 2nd step to the video
WIDTH, HEIGHT = 800, 600
ROBOT_NAMES = ["Robot 1 (blue)", "Robot 2 (orange)"]
ROBOT_COLORS = [[0.1, 0.4, 0.9, 1], [0.95, 0.5, 0.1, 1]]
START_XY_TEAM = [(-0.5, 0.0), (0.5, 0.0)]


def rule_based_action(obs):
    """Backup controller: steer toward the (assigned) weed.
    obs[0], obs[1] = weed offset in the robot's frame (x forward, y left)."""
    wdx, wdy = float(obs[0]), float(obs[1])
    angle = math.atan2(wdy, wdx)
    turn = float(np.clip(2.0 * angle, -1.0, 1.0))
    forward = float(np.clip(1.0 - abs(angle), 0.15, 1.0))
    return np.array([turn, forward], dtype=np.float32)


# ======================================================================
# Simulation world (one PyBullet field, N robots)
# ======================================================================
class MultiRobotField:
    def __init__(self, n_robots=2):
        self.n = n_robots
        self._client = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self._build_field()
        self._build_robots()
        for _ in range(30):
            p.stepSimulation()

        self.step_count = 0
        self.weeds_removed_by = [0] * n_robots
        self.crop_touches = [0] * n_robots
        self.crop_contact_steps = [0] * n_robots
        self._in_contact = [False] * n_robots

        self._view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0], distance=9.5,
            yaw=0, pitch=-80, roll=0, upAxisIndex=2)
        self._proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=50)

    # ---------------- building ----------------
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

        rng = random.Random(RANDOM_SEED)       # same fixed weed layout as before
        self.weed_ids, self.weed_pos = [], {}
        placed = 0
        while placed < NUM_WEEDS:
            x = rng.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
            y = rng.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
            too_close = any((x - cx) ** 2 + (y - cy) ** 2 < 0.35 ** 2
                            for cx, cy in self.crop_positions)
            if not too_close:
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.06,
                                          rgbaColor=[0.55, 0.27, 0.07, 1])
                wid = p.createMultiBody(0, -1, vis, [x, y, 0.06])
                self.weed_ids.append(wid)
                self.weed_pos[wid] = (x, y)
                placed += 1

    def _build_robots(self):
        with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
            f.write(ROBOT_URDF)
            path = f.name
        starts = START_XY_TEAM[:self.n] if self.n > 1 else [(0.0, 0.0)]
        self.robot_ids = []
        for i, (sx, sy) in enumerate(starts):
            rid = p.loadURDF(path, basePosition=[sx, sy, 0.08])
            p.changeVisualShape(rid, -1, rgbaColor=ROBOT_COLORS[i % len(ROBOT_COLORS)])
            self.robot_ids.append(rid)
        os.remove(path)

        # robots may pass through each other (keeps the demo simple)
        if self.n > 1:
            for a in range(-1, 3):
                for b in range(-1, 3):
                    p.setCollisionFilterPair(self.robot_ids[0], self.robot_ids[1], a, b, 0)

    # ---------------- helpers ----------------
    def pose(self, i):
        pos, orn = p.getBasePositionAndOrientation(self.robot_ids[i])
        return pos[0], pos[1], p.getEulerFromQuaternion(orn)[2]

    def _nearest_crop_offset(self, rx, ry):
        best, best_d = None, 1e9
        for cx, cy in self.crop_positions:
            d = math.hypot(cx - rx, cy - ry)
            if d < best_d:
                best, best_d = (cx, cy), d
        return best[0] - rx, best[1] - ry

    def observation(self, i, target_id):
        """Same 5 numbers the PPO model was trained on, but the 'nearest weed'
        is the weed the Planner assigned to this robot."""
        rx, ry, yaw = self.pose(i)
        if target_id in self.weed_pos:
            wdx = self.weed_pos[target_id][0] - rx
            wdy = self.weed_pos[target_id][1] - ry
        else:
            wdx, wdy = 0.0, 0.0
        cdx, cdy = self._nearest_crop_offset(rx, ry)

        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        wdx_r = wdx * cos_y - wdy * sin_y
        wdy_r = wdx * sin_y + wdy * cos_y
        cdx_r = cdx * cos_y - cdy * sin_y
        cdy_r = cdx * sin_y + cdy * cos_y
        frac_left = len(self.weed_ids) / NUM_WEEDS
        return np.array([wdx_r, wdy_r, cdx_r, cdy_r, frac_left], dtype=np.float32)

    # ---------------- stepping ----------------
    def step(self, actions):
        for rid, act in zip(self.robot_ids, actions):
            turn = float(np.clip(act[0], -1, 1))
            forward = float(np.clip(act[1], -1, 1))
            left = (forward - turn) * MAX_SPEED
            right = (forward + turn) * MAX_SPEED
            p.setJointMotorControl2(rid, 0, p.VELOCITY_CONTROL, targetVelocity=-left, force=5)
            p.setJointMotorControl2(rid, 1, p.VELOCITY_CONTROL, targetVelocity=-right, force=5)
        for _ in range(FRAME_SKIP):
            p.stepSimulation()
        self.step_count += 1

        poses = [self.pose(i)[:2] for i in range(self.n)]

        # weed removal (credited to the robot that reached it)
        for wid in list(self.weed_ids):
            wx, wy = self.weed_pos[wid]
            for i, (rx, ry) in enumerate(poses):
                if math.hypot(wx - rx, wy - ry) < WEED_REACH:
                    p.removeBody(wid)
                    self.weed_ids.remove(wid)
                    del self.weed_pos[wid]
                    self.weeds_removed_by[i] += 1
                    break

        # crop contact (count each new contact once, plus total contact steps)
        for i, (rx, ry) in enumerate(poses):
            touching = any(math.hypot(cx - rx, cy - ry) < CROP_HIT_DIST
                           for cx, cy in self.crop_positions)
            if touching:
                self.crop_contact_steps[i] += 1
                if not self._in_contact[i]:
                    self.crop_touches[i] += 1
            self._in_contact[i] = touching

    def render(self):
        _, _, rgb, _, _ = p.getCameraImage(WIDTH, HEIGHT, self._view, self._proj,
                                           renderer=p.ER_TINY_RENDERER)
        return np.array(rgb, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]

    def stats(self):
        removed = NUM_WEEDS - len(self.weed_ids)
        return {
            "n_robots": self.n,
            "steps": self.step_count,
            "weeds_removed": removed,
            "total_weeds": NUM_WEEDS,
            "weeds_left": len(self.weed_ids),
            "completed": len(self.weed_ids) == 0,
            "crop_touches": int(sum(self.crop_touches)),
            "per_robot": [
                {"robot": ROBOT_NAMES[i % len(ROBOT_NAMES)] if self.n > 1 else "Robot",
                 "weeds_removed": self.weeds_removed_by[i],
                 "crop_touches": self.crop_touches[i],
                 "crop_contact_steps": self.crop_contact_steps[i]}
                for i in range(self.n)
            ],
        }

    def close(self):
        if p.isConnected(self._client):
            p.disconnect(self._client)


# ======================================================================
# Agents
# ======================================================================
class PerceptionAgent:
    """Senses the field (simulated sensors)."""
    name = "Perception Agent"

    def __init__(self, field):
        self.field = field

    def sense(self):
        f = self.field
        return {
            "weeds": dict(f.weed_pos),
            "crops": list(f.crop_positions),
            "robots": [f.pose(i) for i in range(f.n)],
        }


class PlannerAgent:
    """Gives each robot its own weed. A robot keeps its target until it is
    removed, then gets the nearest weed nobody else is going for."""
    name = "Planner Agent"

    def __init__(self):
        self.targets = {}

    def assign(self, scene):
        weeds, robots = scene["weeds"], scene["robots"]
        for r in list(self.targets):
            if self.targets[r] not in weeds:
                self.targets[r] = None
        taken = {t for t in self.targets.values() if t is not None}
        for r, (rx, ry, _) in enumerate(robots):
            if self.targets.get(r) is None:
                free = [w for w in weeds if w not in taken] or list(weeds)
                if free:
                    best = min(free, key=lambda w: math.hypot(weeds[w][0] - rx,
                                                              weeds[w][1] - ry))
                    self.targets[r] = best
                    taken.add(best)
        return self.targets


class ControllerAgent:
    """Drives one robot using the trained PPO policy (or rule-based backup)."""

    def __init__(self, index, model=None):
        self.index = index
        self.model = model
        self.name = f"Controller Agent {index + 1}"

    def act(self, obs):
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        return rule_based_action(obs)


# ======================================================================
# One full mission
# ======================================================================
def run_episode(n_robots=2, model=None, record=False):
    field = MultiRobotField(n_robots)
    perception = PerceptionAgent(field)
    planner = PlannerAgent()
    controllers = [ControllerAgent(i, model) for i in range(n_robots)]

    frames = [field.render()] if record else []
    done = False
    while not done:
        scene = perception.sense()
        targets = planner.assign(scene)
        actions = [c.act(field.observation(i, targets.get(i)))
                   for i, c in enumerate(controllers)]
        field.step(actions)
        done = (not field.weed_ids) or field.step_count >= MAX_STEPS
        if record and (field.step_count % RECORD_EVERY == 0 or done):
            frames.append(field.render())

    stats = field.stats()
    field.close()
    return stats, frames
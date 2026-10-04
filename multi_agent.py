"""
Multi-Agent Weeding System
------------------------------------------------------
Agents:
    Perception Agent  -> senses weeds, crops and robot poses in the field
    Planner Agent     -> assigns a different weed to every robot (no overlap)
    Controller Agents -> one per robot; each uses the SAME trained PPO model
                         (falls back to rule-based steering if no model)
    Supervisor Agent  -> a third robot that watches the workers. If a worker
                         is about to hit a crop, drives away from its weed,
                         steers the wrong way or gets stuck, the Supervisor
                         flies to it and overrides the command with a safe,
                         guided one.

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

RECORD_EVERY = 1                      # record every step -> slower, clearer video
WIDTH, HEIGHT = 800, 600
ROBOT_NAMES = ["Robot 1 (blue)", "Robot 2 (orange)"]
ROBOT_COLORS = [[0.1, 0.4, 0.9, 1], [0.95, 0.5, 0.1, 1]]
START_XY_TEAM = [(-0.5, 0.0), (0.5, 0.0)]

SUPERVISOR_COLOR = [0.55, 0.2, 0.8, 1]
SUP_START = (0.0, -1.5)
SUP_HEIGHT = 0.7                      # hovers above the field
SUP_SPEED = 0.18                      # metres per step
SUP_STANDOFF = 0.7                    # how close it hovers to the robot it guides


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rule_based_action(obs):
    """Backup controller: steer toward the (assigned) weed.
    obs[0], obs[1] = weed offset in the robot's frame (x forward, y left)."""
    wdx, wdy = float(obs[0]), float(obs[1])
    angle = math.atan2(wdy, wdx)
    turn = float(np.clip(2.0 * angle, -1.0, 1.0))
    forward = float(np.clip(1.0 - abs(angle), 0.15, 1.0))
    return np.array([turn, forward], dtype=np.float32)


def guided_action(obs):
    """The safe command the Supervisor gives: go to the weed, but keep away from crops."""
    wdx, wdy, cdx, cdy = (float(v) for v in obs[:4])
    dist_w = math.hypot(wdx, wdy)
    if dist_w < 1e-6:
        return np.zeros(2, dtype=np.float32)
    ux, uy = wdx / dist_w, wdy / dist_w
    crop_d = math.hypot(cdx, cdy)
    if 1e-6 < crop_d < 0.8:
        k = (0.8 - crop_d) / 0.8 * 1.5            # stronger push the closer the crop
        ux += k * (-cdx / crop_d)
        uy += k * (-cdy / crop_d)
    angle = math.atan2(uy, ux)
    turn = float(np.clip(2.0 * angle, -1.0, 1.0))
    forward = float(np.clip(1.0 - abs(angle), 0.0, 1.0))
    if abs(angle) > 1.0:
        forward = 0.1                              # turn first, then drive
    if crop_d < 0.5:
        forward *= 0.6                             # slow down near a crop
    return np.array([turn, forward], dtype=np.float32)


def faulty_action(obs):
    """Used only for the optional fault test: steer AWAY from the weed."""
    angle = math.atan2(float(obs[1]), float(obs[0]))
    return np.array([-1.0 if angle >= 0 else 1.0, 0.8], dtype=np.float32)


# ======================================================================
# Simulation world (one PyBullet field, N robots + a flying supervisor)
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
        self.traj = [[] for _ in range(n_robots)]   # per-robot [x, y, yaw] every step
        self.weed_removed = {}                        # weed id -> (step, robot index)

        # supervisor (a flying body, no collisions)
        self.sup_xy = list(SUP_START)
        self.sup_goal = None
        self.sup_traj = []
        self.guided_log = [[0] for _ in range(n_robots)]
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.22, rgbaColor=SUPERVISOR_COLOR)
        self.sup_id = p.createMultiBody(0, -1, vis, [SUP_START[0], SUP_START[1], SUP_HEIGHT])

        self._log_poses()

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
                # drawn big and red so weeds are easy to see in the video
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.14,
                                          rgbaColor=[0.85, 0.12, 0.1, 1])
                wid = p.createMultiBody(0, -1, vis, [x, y, 0.14])
                self.weed_ids.append(wid)
                self.weed_pos[wid] = (x, y)
                placed += 1
        self.weed_init = [(wid, self.weed_pos[wid][0], self.weed_pos[wid][1])
                          for wid in self.weed_ids]

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

    def _log_poses(self):
        for i in range(self.n):
            x, y, yaw = self.pose(i)
            self.traj[i].append([round(x, 3), round(y, 3), round(yaw, 3)])
        self.sup_traj.append([round(self.sup_xy[0], 3), round(self.sup_xy[1], 3)])

    def log_guided(self, flags):
        for i, g in enumerate(flags):
            self.guided_log[i].append(1 if g else 0)

    def _move_supervisor(self):
        if self.sup_goal is None:
            return
        gx, gy = self.sup_goal
        dx, dy = gx - self.sup_xy[0], gy - self.sup_xy[1]
        d = math.hypot(dx, dy)
        if d > SUP_STANDOFF:
            move = min(SUP_SPEED, d - SUP_STANDOFF)
            self.sup_xy[0] += dx / d * move
            self.sup_xy[1] += dy / d * move
            p.resetBasePositionAndOrientation(
                self.sup_id, [self.sup_xy[0], self.sup_xy[1], SUP_HEIGHT], [0, 0, 0, 1])

    def trajectory(self):
        """Everything the web app needs to replay the mission."""
        weeds = []
        for wid, x, y in self.weed_init:
            info = self.weed_removed.get(wid)
            weeds.append({"x": round(x, 3), "y": round(y, 3),
                          "removed_step": info[0] if info else None,
                          "by": info[1] if info else None})
        return {
            "field_size": FIELD_SIZE,
            "weed_reach": WEED_REACH,
            "crops": [[x, y] for x, y in self.crop_positions],
            "weeds": weeds,
            "robots": self.traj,
            "supervisor": self.sup_traj,
            "guided": self.guided_log,
            "names": [ROBOT_NAMES[i % len(ROBOT_NAMES)] for i in range(self.n)],
            "steps": self.step_count,
        }

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
                    self.weed_removed[wid] = (self.step_count, i)
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

        self._move_supervisor()
        self._log_poses()

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


class SupervisorAgent:
    """Watches every worker. When a command looks wrong it replaces it with a
    guided one for a few steps and logs the correction."""
    name = "Supervisor Agent"

    DANGER_DIST = 0.42        # closer than this to a crop, driving at it
    OFF_COURSE_RAD = 1.5      # travelling >86 degrees away from its weed
    WRONG_TURN_ANGLE = 0.6    # weed clearly on one side, robot turns to the other
    STUCK_STEPS = 30          # no progress for this many steps
    STUCK_PROGRESS = 0.05     # metres
    HOLD_STEPS = 12           # keep guiding for this long after a problem
    MERGE_GAP = 25            # problems closer than this are one event

    def __init__(self, n_robots):
        self.n = n_robots
        self.hold = [0] * n_robots
        self.reason = [None] * n_robots
        self.last_target = [None] * n_robots
        self.dist_hist = [[] for _ in range(n_robots)]
        self.events = []
        self._open = [None] * n_robots

    def _check(self, i, obs, action, target_id):
        wdx, wdy, cdx, cdy = (float(v) for v in obs[:4])
        turn, forward = float(action[0]), float(action[1])
        angle = math.atan2(wdy, wdx)
        crop_d = math.hypot(cdx, cdy)

        toward = (forward > 0.2 and cdx > 0) or (forward < -0.2 and cdx < 0)
        if crop_d < self.DANGER_DIST and toward and abs(cdy) < 0.35:
            return "too close to a crop"

        travel_err = abs(_wrap(angle - (0.0 if forward >= 0 else math.pi)))
        if abs(forward) > 0.3 and travel_err > self.OFF_COURSE_RAD:
            return "driving away from its weed"

        if (forward >= 0 and abs(angle) > self.WRONG_TURN_ANGLE
                and abs(turn) > 0.6 and turn * angle < 0):
            return "steering the wrong way"

        dist = math.hypot(wdx, wdy)
        if self.last_target[i] != target_id:
            self.last_target[i] = target_id
            self.dist_hist[i] = []
        h = self.dist_hist[i]
        h.append(dist)
        if len(h) > self.STUCK_STEPS:
            if h[-self.STUCK_STEPS - 1] - dist < self.STUCK_PROGRESS and dist > WEED_REACH:
                self.dist_hist[i] = []
                return "no progress toward its weed"
            h.pop(0)
        return None

    def review(self, step, i, obs, action, target_id, injected=False):
        """Returns (action_to_use, guided?, reason)."""
        new_problem = False
        if self.hold[i] > 0:
            self.hold[i] -= 1
            reason = self.reason[i]
        else:
            reason = self._check(i, obs, action, target_id)
            if reason:
                self.hold[i] = self.HOLD_STEPS
                self.reason[i] = reason
                new_problem = True
        if not reason:
            return action, False, None

        ev = self._open[i]
        if new_problem and (ev is None or ev["reason"] != reason
                            or step - ev["end"] > self.MERGE_GAP):
            ev = {"robot": i, "robot_name": ROBOT_NAMES[i % len(ROBOT_NAMES)],
                  "start": step, "end": step, "reason": reason,
                  "injected": bool(injected)}
            self.events.append(ev)
            self._open[i] = ev
        elif ev is not None:
            ev["end"] = step
            ev["injected"] = ev["injected"] or bool(injected)
        return guided_action(obs), True, reason

    def summary(self, fault):
        return {"interventions": len(self.events), "events": self.events[:20],
                "fault_injected": bool(fault), "fault": fault}


# ======================================================================
# Video overlay
# ======================================================================
def _annotate(frame, text, color):
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=20)
        except TypeError:
            font = ImageFont.load_default()
        d.rectangle([0, 0, WIDTH, 34], fill=color)
        d.text((10, 6), text, fill=(255, 255, 255), font=font)
        return np.array(img)
    except Exception:
        return frame


# ======================================================================
# One full mission
# ======================================================================
def run_episode(n_robots=2, model=None, record=False, fault=None):
    """fault (optional): {"robot": 1, "start": 40, "end": 75} -> that robot's
    steering is deliberately corrupted so the Supervisor has something to fix."""
    field = MultiRobotField(n_robots)
    perception = PerceptionAgent(field)
    planner = PlannerAgent()
    controllers = [ControllerAgent(i, model) for i in range(n_robots)]
    supervisor = SupervisorAgent(n_robots) if n_robots > 1 else None

    def banner():
        if supervisor:
            for i in range(n_robots):
                if field.guided_log[i][-1]:
                    return (f"SUPERVISOR guiding {ROBOT_NAMES[i]}: {supervisor.reason[i]}",
                            (123, 63, 196))
        return (f"Step {field.step_count} | weeds left {len(field.weed_ids)} | "
                "supervisor: all robots OK", (46, 125, 50))

    def snap():
        text, color = banner()
        return _annotate(field.render(), text, color)

    frames = [snap()] if record else []
    done = False
    while not done:
        scene = perception.sense()
        targets = planner.assign(scene)
        actions, guided = [], [False] * n_robots
        for i, c in enumerate(controllers):
            tid = targets.get(i)
            obs = field.observation(i, tid)
            act = c.act(obs)
            injected = bool(fault and fault["robot"] == i
                            and fault["start"] <= field.step_count < fault["end"])
            if injected:
                act = faulty_action(obs)
            if supervisor:
                act, guided[i], _ = supervisor.review(
                    field.step_count + 1, i, obs, act, tid, injected)
            actions.append(act)

        # where the supervisor flies next: to a robot it is guiding, else the middle
        if supervisor:
            if any(guided):
                gx, gy, _ = scene["robots"][guided.index(True)]
            else:
                gx = sum(r[0] for r in scene["robots"]) / n_robots
                gy = sum(r[1] for r in scene["robots"]) / n_robots
            field.sup_goal = (gx, gy)

        field.step(actions)
        field.log_guided(guided)
        done = (not field.weed_ids) or field.step_count >= MAX_STEPS
        if record and (field.step_count % RECORD_EVERY == 0 or done):
            frames.append(snap())

    stats = field.stats()
    traj = field.trajectory()
    traj["events"] = supervisor.events if supervisor else []
    traj["fault"] = fault
    stats["trajectory"] = traj
    if supervisor:
        stats["supervisor"] = supervisor.summary(fault)
    field.close()
    return stats, frames
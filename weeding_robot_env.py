"""
Autonomous Weeding Robot - PyBullet simulation
------------------------------------------------
The robot finds the nearest weed, drives to it, removes it, and repeats.

Extras in this version (all simple):
  - Settings at the top (easy to change)
  - Live score panel inside the window
  - Red line to the current target weed
  - Blue trail showing the robot's path
  - Stuck recovery (robot backs up and turns if it gets jammed)
  - Reward tracking (ready for the Reinforcement Learning step later)

  
Run:
    python weeding_robot_env.py
"""

import math
import time
import random
import tempfile

import pybullet as p
import pybullet_data


# ---------------------------------------------------------------------
# 0. Settings (change these to experiment)
# ---------------------------------------------------------------------
FIELD_SIZE = 4            # field goes from -FIELD_SIZE to +FIELD_SIZE
ROWS, COLS = 4, 4         # crop grid
NUM_WEEDS = 12
RANDOM_SEED = 42          # same seed = same weed positions every run

WEED_REACH = 0.30         # remove weed when robot is closer than this (m)
CROP_AVOID_DIST = 0.7     # start steering away from a crop inside this (m)
MAX_SPEED = 10            # max wheel speed (rad/s)
MAX_STEPS_PER_WEED = 4000 # give up on a weed after this many steps

REWARD_WEED = 10          # +10 for every weed removed
REWARD_CROP_HIT = -5      # -5 every time the robot touches a crop
REWARD_STEP = -0.01       # tiny penalty per step (encourages speed)

STEP_SLEEP = 1 / 240      # real-time speed. Use 0 for a fast simulation.


# ---------------------------------------------------------------------
# 1. Basic setup
# ---------------------------------------------------------------------
p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane_id = p.loadURDF("plane.urdf")

# camera: look at the whole field from above
p.resetDebugVisualizerCamera(cameraDistance=9, cameraYaw=0,
                             cameraPitch=-65, cameraTargetPosition=[0, 0, 0])


# ---------------------------------------------------------------------
# 2. Robot: box chassis + two driven wheels + front caster
# ---------------------------------------------------------------------
ROBOT_URDF = """
<robot name="weeder_robot">

  <link name="chassis">
    <visual>
      <geometry><box size="0.35 0.25 0.12"/></geometry>
      <material name="grey"><color rgba="0.3 0.3 0.3 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.35 0.25 0.12"/></geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <link name="left_wheel">
    <visual>
      <geometry><cylinder radius="0.06" length="0.04"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <geometry><cylinder radius="0.06" length="0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <link name="right_wheel">
    <visual>
      <geometry><cylinder radius="0.06" length="0.04"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <geometry><cylinder radius="0.06" length="0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <link name="caster_wheel">
    <visual>
      <geometry><sphere radius="0.04"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="chassis"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.15 -0.02" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="chassis"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.02" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="caster_joint" type="fixed">
    <parent link="chassis"/>
    <child link="caster_wheel"/>
    <origin xyz="0.15 0 -0.04" rpy="0 0 0"/>
  </joint>

</robot>
"""

with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
    f.write(ROBOT_URDF)
    robot_urdf_path = f.name

robot_id = p.loadURDF(robot_urdf_path, basePosition=[0, 0, 0.08])

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1


def drive(left_fwd, right_fwd):
    """
    Drive the wheels. Inputs are 'forward' wheel speeds (rad/s).
    With this wheel orientation a NEGATIVE joint velocity moves the
    robot forward (toward the caster side), so the sign is flipped here.
    """
    left_fwd = max(-MAX_SPEED, min(MAX_SPEED, left_fwd))
    right_fwd = max(-MAX_SPEED, min(MAX_SPEED, right_fwd))
    p.setJointMotorControl2(robot_id, LEFT_WHEEL_JOINT, p.VELOCITY_CONTROL,
                            targetVelocity=-left_fwd, force=5)
    p.setJointMotorControl2(robot_id, RIGHT_WHEEL_JOINT, p.VELOCITY_CONTROL,
                            targetVelocity=-right_fwd, force=5)


# ---------------------------------------------------------------------
# 3. Build the field: crops in a neat grid, weeds scattered randomly
# ---------------------------------------------------------------------
crop_ids = []
weed_ids = []
weed_pos = {}             # weed body id -> (x, y)


def make_crop(x, y):
    """A green cylinder = one crop plant."""
    col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=0.3)
    vis_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.08, length=0.3, rgbaColor=[0.15, 0.6, 0.15, 1]
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=[x, y, 0.15],
    )


def make_weed(x, y):
    """A small brown sphere = one weed. No collision, so the robot
    can drive over it and 'pull' it without getting stuck."""
    vis_shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.06, rgbaColor=[0.55, 0.27, 0.07, 1]
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis_shape,
        basePosition=[x, y, 0.06],
    )


crop_positions = []
xs = [(-FIELD_SIZE + 1) + i * 2 for i in range(COLS)]
ys = [(-FIELD_SIZE + 1) + j * 2 for j in range(ROWS)]
for x in xs:
    for y in ys:
        crop_positions.append((x, y))
        crop_ids.append(make_crop(x, y))

random.seed(RANDOM_SEED)
placed = 0
while placed < NUM_WEEDS:
    x = random.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
    y = random.uniform(-FIELD_SIZE + 0.3, FIELD_SIZE - 0.3)
    too_close = any((x - cx) ** 2 + (y - cy) ** 2 < 0.35 ** 2 for cx, cy in crop_positions)
    if not too_close:
        wid = make_weed(x, y)
        weed_ids.append(wid)
        weed_pos[wid] = (x, y)
        placed += 1

print(f"Field ready: {len(crop_ids)} crops, {len(weed_ids)} weeds.")


# ---------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------
def wrap_angle(a):
    """Keep an angle between -pi and +pi."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def robot_pose():
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    yaw = p.getEulerFromQuaternion(orn)[2]
    return pos[0], pos[1], yaw


def nearest_weed(rx, ry):
    best, best_d = None, 1e9
    for wid in weed_ids:
        wx, wy = weed_pos[wid]
        d = math.hypot(wx - rx, wy - ry)
        if d < best_d:
            best, best_d = wid, d
    return best


def steer_to(target_x, target_y):
    """Set wheel speeds to move toward the target while staying
    away from nearby crops."""
    rx, ry, yaw = robot_pose()

    # direction to the target
    vx, vy = target_x - rx, target_y - ry
    dist = math.hypot(vx, vy)
    if dist > 1e-6:
        vx, vy = vx / dist, vy / dist

    # push away from crops that are too close
    for cx, cy in crop_positions:
        dx, dy = rx - cx, ry - cy
        d = math.hypot(dx, dy)
        if 1e-6 < d < CROP_AVOID_DIST:
            strength = (CROP_AVOID_DIST - d) / CROP_AVOID_DIST
            vx += 2.5 * strength * dx / d
            vy += 2.5 * strength * dy / d

    err = wrap_angle(math.atan2(vy, vx) - yaw)

    turn = 5.0 * err                               # rotate toward the target
    forward = 9.0 * max(0.0, math.cos(err)) ** 2   # slow down if not facing it
    if abs(err) > 1.0:
        forward = 0.0                              # big error: turn on the spot

    drive(forward - turn, forward + turn)


def step_sim():
    p.stepSimulation()
    if STEP_SLEEP > 0:
        time.sleep(STEP_SLEEP)


def recover_from_stuck():
    """Back up while turning a little, then let the controller try again."""
    print("  Robot looks stuck, backing up...")
    turn_dir = random.choice([-1, 1])
    for _ in range(80):
        if not p.isConnected():
            return
        drive(-6 + 3 * turn_dir, -6 - 3 * turn_dir)
        step_sim()


# ---------------------------------------------------------------------
# 5. On-screen extras: score panel, target line, path trail
# ---------------------------------------------------------------------
hud_id = -1
target_line_id = -1


def update_hud(removed, crop_hits, reward):
    global hud_id
    text = (f"Weeds removed: {removed}/{NUM_WEEDS}   "
            f"Crop touches: {crop_hits}   Reward: {reward:.1f}")
    hud_id = p.addUserDebugText(text, [-4, 4.8, 0.5], textColorRGB=[1, 1, 1],
                                textSize=1.6, replaceItemUniqueId=hud_id)


# ---------------------------------------------------------------------
# 6. Main loop: weed the whole field
# ---------------------------------------------------------------------
for _ in range(120):          # let the robot settle on the ground
    p.stepSimulation()

removed = 0
skipped = 0
crop_hits = 0
touching_now = set()
total_reward = 0.0
total_steps = 0

trail_last = robot_pose()[:2]
update_hud(removed, crop_hits, total_reward)
print("Weeding started...")

while weed_ids and p.isConnected():
    rx, ry, _ = robot_pose()
    target = nearest_weed(rx, ry)
    tx, ty = weed_pos[target]

    steps = 0
    check_pos = (rx, ry)
    reached = False

    while p.isConnected() and steps < MAX_STEPS_PER_WEED:
        rx, ry, _ = robot_pose()

        # close enough -> remove the weed
        if math.hypot(tx - rx, ty - ry) < WEED_REACH:
            p.removeBody(target)
            weed_ids.remove(target)
            removed += 1
            total_reward += REWARD_WEED
            print(f"Weed removed ({removed}/{NUM_WEEDS}) at ({tx:.2f}, {ty:.2f})")
            update_hud(removed, crop_hits, total_reward)
            reached = True
            break

        steer_to(tx, ty)
        step_sim()
        steps += 1
        total_steps += 1
        total_reward += REWARD_STEP

        # red line from the robot to its current target
        target_line_id = p.addUserDebugLine([rx, ry, 0.1], [tx, ty, 0.1],
                                            [1, 0, 0], lineWidth=2,
                                            replaceItemUniqueId=target_line_id)

        # blue trail of where the robot has been
        if steps % 15 == 0:
            p.addUserDebugLine([trail_last[0], trail_last[1], 0.01],
                               [rx, ry, 0.01], [0.2, 0.4, 1], lineWidth=2)
            trail_last = (rx, ry)

        # count every new touch between the robot and a crop
        now = {c for c in crop_ids if p.getContactPoints(robot_id, c)}
        new_hits = len(now - touching_now)
        if new_hits:
            crop_hits += new_hits
            total_reward += REWARD_CROP_HIT * new_hits
            update_hud(removed, crop_hits, total_reward)
        touching_now = now

        # stuck check: every 240 steps, did the robot actually move?
        if steps % 240 == 0:
            if math.hypot(rx - check_pos[0], ry - check_pos[1]) < 0.08:
                recover_from_stuck()
            check_pos = (rx, ry)

    # remove the red line once this weed is finished
    if target_line_id >= 0:
        p.removeUserDebugItem(target_line_id)
        target_line_id = -1

    # timed out: skip this weed so the robot does not loop forever
    if not reached and p.isConnected():
        print(f"Could not reach weed at ({tx:.2f}, {ty:.2f}), skipping it.")
        weed_ids.remove(target)
        p.removeBody(target)
        skipped += 1

if p.isConnected():
    drive(0, 0)
    update_hud(removed, crop_hits, total_reward)
    print("-" * 44)
    print(f"Done in {total_steps} steps")
    print(f"Weeds removed : {removed}")
    print(f"Weeds skipped : {skipped}")
    print(f"Crop touches  : {crop_hits}")
    print(f"Total reward  : {total_reward:.1f}")
    print("Close the PyBullet window to exit.")

# keep the window open so you can look around with the mouse
while p.isConnected():
    p.stepSimulation()
    time.sleep(1 / 240)

# 🌱 AgriBot — Multi-Agent Autonomous Weeding Robots

A simulation-based robotics project where two robots learn, through
**Reinforcement Learning**, to find and remove weeds from a field —
without harming the crops — and coordinate with each other to clear
the field faster than a single robot could alone.

**Live demo:** https://agribot-multi-agent-weeding.streamlit.app

---

## 📌 Overview

Weeds compete with crops for water, sunlight, and nutrients. Removing
them by hand is slow and labour-intensive, and traditional farm robots
rely on fixed, rule-based logic that can't adapt to a changing field.

AgriBot builds a robot that **learns** — through trial and reward — how
to tell crops and weeds apart and clear a field on its own. The project
then extends this into a **multi-agent** system (two robots, each
responsible for one half of the field) with basic **automation**
(the mission runs start-to-finish with no manual input, and a report +
email alert are generated automatically when it finishes).

## 🧩 Problem Statement

- Manual weeding is slow and costly on large farms
- Rule-based weeding robots can't adapt when the field looks different than expected
- Simple robots often can't reliably tell a crop apart from a weed, risking crop damage
- A smarter, learning-based robot is needed

## 🎯 Objectives

1. Build a simulated robot that can navigate a field and tell crops apart from weeds
2. Train the robot with Reinforcement Learning (PPO) so it learns to remove weeds while keeping crops safe

## 🛠️ How It's Solved

| Step | What happens |
|---|---|
| 1 | A virtual field is built in **PyBullet** — a robot, crop plants (green), and weeds (brown) |
| 2 | The robot observes the position of the nearest weed and nearest crop |
| 3 | A reward system is defined: `+10` for removing a weed, `−15` for touching a crop, `−0.1` per step |
| 4 | The robot is trained with **PPO** (via Stable-Baselines3) until it learns an efficient policy |
| 5 | The trained robot is tested and compared against a simple rule-based baseline |
| 6 | The single-robot idea is extended to **two robots**, each working its own half of the field |

## 🧠 How the Agents Work Together

| Agent | Role |
|---|---|
| 👁️ Perception | Senses weed, crop, and robot positions |
| 🧭 Planner | Splits the field so each robot gets its own zone — no overlap |
| 🤖 Controller ×2 | Drives each robot (rule-based steering locally; PPO-trained policy for the core single-robot task) |
| 📝 Report | Summarizes the mission (weeds removed, steps taken, crop touches) |
| 📧 Alert | Sends an email when a mission completes or something goes wrong (e.g. a crop touch) |

## ⚙️ Parameters & Threshold Values

| Parameter | Value |
|---|---|
| Learning rate | 0.0003 |
| Discount factor (gamma) | 0.99 |
| Reward — weed removed | +10 |
| Reward — crop touched | −15 |
| Step penalty | −0.1 |
| Distance threshold to remove a weed | 0.5 m |
| Max steps per episode | 500 |

## 🤖 Robot Design (URDF)

- **Configuration:** differential drive — two motor-driven wheels at the back, one free-rolling caster wheel at the front
- **Wheel joints:** `continuous` (revolute, 0°–360° rotation) for driving and turning
- **Caster joint:** `fixed` — balance only, no motion
- Built directly from an inline URDF string (no external file needed)

## 📁 Project Structure

```
.
├── app.py                   # Streamlit web app (the live demo UI)
├── run_mission.py           # Runs a full 2-robot mission, saves video + stats + report
├── multi_agent.py           # Multi-agent (zone-divided) simulation logic
├── business_automation.py   # Report generation + email alerts
├── weeding_robot_env.py     # Single-robot rule-based baseline (PyBullet)
├── weeding_gym_env.py       # Gymnasium environment wrapper (for RL training)
├── train_ppo.py             # Trains the robot with PPO (Stable-Baselines3)
├── test_trained.py          # Loads the trained model and watches it work (GUI)
├── weeding_ppo_model.zip    # The trained PPO model
├── demo.mp4                 # Recorded mission video (shown in the app)
├── stats.json               # Mission results that feed the app's numbers
├── report.md                # Auto-generated mission report
├── requirements.txt         # Python dependencies
├── packages.txt             # System libraries needed by PyBullet on Streamlit Cloud
└── runtime.txt / .python-version   # Pins the Python version for deployment
```

## 💻 Running It Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

**Watch the rule-based robot work (GUI window):**
```bash
python weeding_robot_env.py
```

**Train the robot from scratch with PPO:**
```bash
python train_ppo.py
```

**Watch the trained PPO robot work:**
```bash
python test_trained.py
```

**Run a full 2-robot mission and generate the video/report:**
```bash
python run_mission.py --no-email
```

**Launch the web app locally:**
```bash
streamlit run app.py
```

## ☁️ Deployment

Deployed for free on **Streamlit Community Cloud**, pulling directly
from this GitHub repository. Python version is pinned to **3.11** (via
`runtime.txt` / `.python-version`) since `pybullet` needs a
pre-built wheel that isn't yet available for newer Python versions.

## 📊 Example Result

- **12/12** weeds removed
- **0** crop touches
- **2 robots** finish faster than 1 robot working the whole field alone

## ⚠️ Limitations

- This is a **simulation** (PyBullet), not a real physical robot
- The field layout is fixed (same crop/weed positions every run)
- The live in-browser demo uses a lightweight rule-based controller for
  smooth real-time performance — the full PPO-trained policy is used
  for the single-robot training/evaluation shown in `test_trained.py`
- Business impact figures (labour cost, time saved) are estimates based on stated assumptions, not measured data

## 🔮 Possible Future Work

- Multi-agent Reinforcement Learning (MARL, e.g. with PettingZoo) so both robots learn jointly instead of using rule-based coordination
- Transfer the trained policy to a real robot for field testing
- Camera-based (vision) weed/crop classification instead of simulated ground-truth positions

## 👩‍💻 Author

**Fatima Mahmood**
Registration No: 24-NTU-CS-FL-1257
BSAI, National Textile University, Faisalabad

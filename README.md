# Multi-Agent Autonomous Weeding Robots

Two PPO-trained robots, coordinated by a Planner agent, remove weeds from a fixed
field of crops in a PyBullet simulation. A GenAI agent (Gemini) writes a farm report
and email alerts are sent automatically.

**Pipeline:** Perception Agent -> Planner Agent -> 2 Controller Agents (PPO) -> Mission Stats -> Gemini Report -> Email Alerts

## Files
- `weeding_gym_env.py` - Gymnasium environment (single robot, used for PPO training)
- `train_ppo.py` - trains the PPO model (`weeding_ppo_model.zip`)
- `multi_agent.py` - Perception, Planner and Controller agents + 2-robot field
- `business_automation.py` - business metrics, Gemini report, Gmail alerts
- `run_mission.py` - runs the whole pipeline, writes `demo.mp4`, `stats.json`, `report.md`
- `app.py` - Streamlit app that shows the recorded run

## Run locally
```
pip install -r requirements-local.txt
python train_ppo.py          # optional, a trained model is included
python run_mission.py --no-email
streamlit run app.py
```
Copy `.env.example` to `.env` and add your own keys (`GEMINI_API_KEY`, `GMAIL_ADDRESS`,
`GMAIL_APP_PASSWORD`, `ALERT_TO`). Never commit `.env`.

## Limitations
- Simulation only (no real robot). The field layout is fixed (seed 42) and the model was trained on it.
- 2 robots finished in 242 steps vs 263 for one robot (about 8% faster).
- Business figures (labour cost, crop loss) are estimates based on stated assumptions.

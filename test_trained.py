"""
Watch the TRAINED robot work (loads weeding_ppo_model.zip)
---------------------------------------------------------------
Run:
    python test_trained.py
"""

import time
from stable_baselines3 import PPO
from weeding_gym_env import WeedingEnv

env = WeedingEnv(render_mode="human")   # GUI window this time
model = PPO.load("weeding_ppo_model")

obs, _ = env.reset()
total_reward = 0.0

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    time.sleep(1 / 240)
    if terminated or truncated:
        print(f"Episode finished. Total reward: {total_reward:.1f}")
        obs, _ = env.reset()
        total_reward = 0.0

env.close()

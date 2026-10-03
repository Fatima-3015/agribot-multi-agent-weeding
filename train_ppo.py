"""
Train the Weeding Robot with PPO (Reinforcement Learning)
--------------------------------------------------------------
Uses the parameters from the project presentation:
    learning_rate = 0.0003
    gamma         = 0.99

Install:
    pip install stable-baselines3 gymnasium pybullet numpy

Run:
    python train_ppo.py

Output:
    weeding_ppo_model.zip   (the trained "brain" of the robot)
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from weeding_gym_env import WeedingEnv

TOTAL_TIMESTEPS = 500_000   # increase to 1_000_000 if reward is still rising at the end


def main():
    env = WeedingEnv(render_mode="none")   # no GUI while training -> much faster
    check_env(env, warn=True)              # sanity-check the environment once

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        gamma=0.99,
        verbose=1,
    )

    print(f"Training PPO for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save("weeding_ppo_model")
    print("Saved trained model to weeding_ppo_model.zip")

    env.close()


if __name__ == "__main__":
    main()
    
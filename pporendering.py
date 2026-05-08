from stable_baselines3 import PPO
from strikerenvironment import StrikerEnv
environment = StrikerEnv()
model = PPO.load("ppo_striker_model")
observation, info = environment.reset()
for step in range(1000):
    action, states = model.predict(
        observation,
        deterministic=True
    )
    observation, reward, terminated, truncated, info = environment.step(action)
    environment.render()
    if terminated or truncated:
        observation, info = environment.reset()
environment.close()

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from strikerenvironment import StrikerEnv
import numpy as np
np.random.seed(64)
environment = StrikerEnv()
environment.reset(seed=64)
check_env(environment) #checking that my custom environment follows gymnasium structure

model = PPO("MlpPolicy", environment, seed=64, policy_kwargs=dict(net_arch=[64, 64]), verbose=1)
model.learn(total_timesteps=200000)
model.save("ppo_striker_model")
environment.close()
print("PPO training completed, model updated/saved")
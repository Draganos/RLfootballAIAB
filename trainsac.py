from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from strikerenvironment import StrikerEnv
environment = StrikerEnv()
check_env(environment) #for gymnasium
model = SAC(
    "MlpPolicy",
    environment,
    policy_kwargs=dict(
        net_arch=[64, 64]
    ),
    verbose=1
)

model.learn(
    total_timesteps=50000
)

model.save("sac_striker_model")

environment.close()

print("SAC training completed and model saved")
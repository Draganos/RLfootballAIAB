from stable_baselines3 import SAC
from strikerenvironment import StrikerEnv
environment = StrikerEnv()
model = SAC.load("sac_striker_model")
sensorvector, info = environment.reset()
for step in range(1000):
    movementcontroller, states = model.predict(
        sensorvector,
        deterministic=True
    )
    sensorvector, reward, terminated, truncated, info = environment.step(
        movementcontroller
    )
    environment.render()
    if terminated or truncated:
        sensorvector, info = environment.reset()
environment.close()
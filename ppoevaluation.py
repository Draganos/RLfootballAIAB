from stable_baselines3 import PPO
from strikerenvironment import StrikerEnv
import numpy as np


environment = StrikerEnv()
model = PPO.load("ppo_striker_model")
renders = 20
successfulgoals = 0
totalrewards = []
renderinglengths = []


for episode in range(renders):

    sensorvector, info = environment.reset()

    perrenderreward = 0
    steps = 0
    done = False

    while not done:
        movementcontroller, states = model.predict(
            sensorvector,
            deterministic=True
        )
        sensorvector, reward, terminated, truncated, info = environment.step(
            movementcontroller
        )
        perrenderreward += reward
        steps += 1
        if info["goalscored"]:
            successfulgoals += 1

        done = terminated or truncated

    totalrewards.append(perrenderreward)
    renderinglengths.append(steps)


successrate = successfulgoals / renders
print("PPO Evaluation Results")
print("//// VV ////")
print("Amount of Renders:", renders)
print("Success rate:", successrate)
print("Average reward:", np.mean(totalrewards))
print("Average rendering (episode) length:", np.mean(renderinglengths))

environment.close()
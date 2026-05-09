from stable_baselines3 import PPO, SAC
from strikerenvironment import StrikerEnv
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(64)

def applygaussiannoise(sensorvector, noiselevel): #only applies noise to spatial sensors (indices 0-5). Indices 6-7 velocity/angular velocity NOT spatial sensors
    noisysensorvector = sensorvector.copy()
    noise = np.random.normal(0, noiselevel, size=6).astype(np.float32)
    noisysensorvector[:6] += noise
    return noisysensorvector

def runnoiseevaluation(model, modelname, noiselevel, renders=100):
    environment = StrikerEnv()
    environment.reset(seed=64)
    successfulgoals = 0
    episoderewards = []
    episodelengths = []

    for episode in range(renders):
        sensorvector, info = environment.reset()
        episodereward = 0
        steps = 0
        done = False

        while not done:
            noisysensorvector = applygaussiannoise(sensorvector, noiselevel)
            movementcontroller, states = model.predict(
                noisysensorvector,
                deterministic=True
            )
            sensorvector, reward, terminated, truncated, info = environment.step(
                movementcontroller
            )
            episodereward += reward
            steps += 1
            if info["goalscored"]:
                successfulgoals += 1
            done = terminated or truncated

        episoderewards.append(episodereward)
        episodelengths.append(steps)

    environment.close()

    successrate = successfulgoals / renders
    successstd = np.std(
        [1 if i < successfulgoals else 0 for i in range(renders)]
    )
    avgreward = np.mean(episoderewards)
    avglength = np.mean(episodelengths)
    stdlength = np.std(episodelengths)

    print(f"{modelname} | Noise: {noiselevel:.2f} | "
          f"Success: {successrate:.2f} | "
          f"Avg reward: {avgreward:.2f} | "
          f"Avg length: {avglength:.1f} ± {stdlength:.1f}")

    return successrate, successstd, avgreward, avglength, stdlength


ppo_model = PPO.load("ppo_striker_model")
sac_model = SAC.load("sac_striker_model")

noiselevels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]

ppo_successrates, ppo_successstds = [], []
sac_successrates, sac_successstds = [], []
ppo_lengths, ppo_lengthstds = [], []
sac_lengths, sac_lengthstds = [], []

for noiselevel in noiselevels:
    ppo_sr, ppo_sstd, _, ppo_len, ppo_lstd = runnoiseevaluation(
        ppo_model, "PPO", noiselevel
    )
    sac_sr, sac_sstd, _, sac_len, sac_lstd = runnoiseevaluation(
        sac_model, "SAC", noiselevel
    )
    ppo_successrates.append(ppo_sr)
    ppo_successstds.append(ppo_sstd)
    sac_successrates.append(sac_sr)
    sac_successstds.append(sac_sstd)
    ppo_lengths.append(ppo_len)
    ppo_lengthstds.append(ppo_lstd)
    sac_lengths.append(sac_len)
    sac_lengthstds.append(sac_lstd)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#using matplotlib applying plots
#plot 1: Success rate with error bars

ax1.errorbar(noiselevels, ppo_successrates, yerr=ppo_successstds,
             marker="o", label="PPO", capsize=4)
ax1.errorbar(noiselevels, sac_successrates, yerr=sac_successstds,
             marker="s", label="SAC", capsize=4)
ax1.set_xlabel("Gaussian Noise Std (σ)")
ax1.set_ylabel("Goal Scoring Rate")
ax1.set_title("Success Rate vs Sensor Noise")
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 1.05)

#plot 2: Average length of render (episode) with error bars
ax2.errorbar(noiselevels, ppo_lengths, yerr=ppo_lengthstds,
             marker="o", label="PPO", capsize=4)
ax2.errorbar(noiselevels, sac_lengths, yerr=sac_lengthstds,
             marker="s", label="SAC", capsize=4)
ax2.set_xlabel("Gaussian Noise Std (σ)")
ax2.set_ylabel("Average Render Length (steps)")
ax2.set_title("Episode Length vs Sensor Noise")
ax2.legend()
ax2.grid(True)

plt.suptitle("PPO vs SAC Robustness Under Sensor Noise", fontsize=13)
plt.tight_layout()
plt.savefig("noise_robustness_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved to noise_robustness_results.png")
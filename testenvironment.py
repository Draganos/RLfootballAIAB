from strikerenvironment import StrikerEnv
environment = StrikerEnv()
observation, info = environment.reset()
for step in range(500):
    action = environment.action_space.sample()
    observation, reward, terminated, truncated, info = environment.step(action)
    environment.render() #rendering pygame animation from strikerevnironment.py
    print("Step:", step)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Max Timeout (truncated):", truncated)
    print()

    if terminated or truncated:
        observation, info = environment.reset()
environment.close()
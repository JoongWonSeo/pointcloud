import gymnasium as gym
import robosuite_envs

from stable_baselines3 import SAC

task = 'RobosuiteReach-v0'

train = True
if train:
    env = gym.make(task)

    model = SAC("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4, progress_bar=True)
    model.save(task)

    del model # remove to demonstrate saving and loading
    env.close()

model = SAC.load(task)

env = gym.make(task, render_mode='human')
obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      obs, info = env.reset()
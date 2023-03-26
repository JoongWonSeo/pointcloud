import gymnasium as gym
import gymnasium_robotics
import robosuite_envs
import pointcloud_vision

from stable_baselines3 import SAC, HerReplayBuffer

# task = 'RobosuiteReach-v0'
# task = 'RobosuitePickAndPlace-v0'
task = 'FetchPickAndPlace-v2'

train = False
if train:
    env = gym.make(task)

    model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, verbose=1)
    model.learn(total_timesteps=500000, log_interval=4, progress_bar=True)
    model.save(task)

    del model # remove to demonstrate saving and loading
    env.close()


env = gym.make(task.replace('Robosuite', 'Vision'), render_mode='human', max_episode_steps=100)
model = SAC.load('weights/'+task, env=env)
obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print('success' if info['is_success'] else 'failure')
        obs, info = env.reset()
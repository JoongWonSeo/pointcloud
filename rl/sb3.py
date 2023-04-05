import numpy as np
import gymnasium as gym
import gymnasium_robotics
import robosuite_envs
import pointcloud_vision

from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MultiInputPolicy

# task = 'GTReach-v0'
task = 'VisionReach-v0'
agent = 'VisionReach-v0'
# task = agent = 'RobosuitePickAndPlace-v0'

env = gym.make(task, render_mode='human', max_episode_steps=50)
# env = DummyVecEnv([lambda: gym.make(task, render_mode='human', max_episode_steps=100)])
# env = VecNormalize.load('weights/'+task+'.zip', env)
#  do not update them at test time
# env.training = False
# # reward normalization is not needed at test time
# env.norm_reward = False

if True:
    model = TQC.load('weights/'+task, env=env)
    model.policy.save('weights/'+task+'_policy')
    policy = model.policy
else:
    policy = MultiInputPolicy.load('weights/'+agent+'_policy')


obs, info = env.reset()

ep_reward = 0
while True:
    action, _states = policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    ep_reward += bool(np.linalg.norm(env.episode_goal_state['robot0_eef_pos'] - env.raw_state['robot0_eef_pos']) < 0.05) - 1
    env.render()
    if terminated or truncated:
        print(('success' if info['is_success'] else 'failure')+' reward: '+str(ep_reward))  
        obs, info = env.reset()
        ep_reward = 0


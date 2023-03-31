import gymnasium as gym
import gymnasium_robotics
import robosuite_envs
import pointcloud_vision

from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MultiInputPolicy

task = 'RobosuiteReach-v0'

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
    policy = MultiInputPolicy.load('weights/'+task+'_policy')


obs, info = env.reset()

while True:
    action, _states = policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print('success' if info['is_success'] else 'failure')
        obs, info = env.reset()
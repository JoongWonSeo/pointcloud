import numpy as np
import gymnasium as gym
import gymnasium_robotics
import robosuite_envs
import pointcloud_vision
from robosuite_envs.encoders import PassthroughEncoder

from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MultiInputPolicy

# task, agent = 'GTReach-v0', 'RoboReach-v0'
# task, agent = 'VisionPickAndPlace-v0', 'RoboPickAndPlace-v0'
task, agent = 'VisionPushMultiSeg-v0', 'RoboPush-v0'

env = gym.make(task, render_mode='human', max_episode_steps=50)
gt = PassthroughEncoder(env=env, obs_keys=env.encoder.obs_keys, goal_keys=env.encoder.goal_keys)

if False:
    model = TQC.load('weights/'+agent, env=env)
    model.policy.save('weights/'+agent+'_policy')
    policy = model.policy
else:
    policy = MultiInputPolicy.load('weights/'+agent+'_policy')


obs, info = env.reset()

ep_reward = 0
while True:
    action, _states = policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    gt_goal = gt.encode_goal(env.goal_state)
    gt_achieved = gt.encode_goal(env.raw_state)
    ep_reward += bool(env.check_success(gt_achieved, gt_goal, info=info, force_gt=True)) - 1
    env.render()
    if terminated or truncated:
        print(('success' if info['is_success'] else 'failure')+' reward: '+str(ep_reward))  
        obs, info = env.reset()
        ep_reward = 0


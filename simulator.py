# This assumes the gym environment is a GoalEnv as defined by gymnasium_robotics.

import numpy as np
import torch
import gymnasium as gym
import gym_robosuite_envs
from utils import *
import rl_core as core

# global variables
horizon = 50

# setup environment and agent
# env = gym.make('FetchReach-v3', render_mode='human', max_episode_steps=horizon)
env = gym.make('RobosuiteReach-v0', render_mode='human', max_episode_steps=horizon)

agent_input_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
agent_output_dim = env.action_space.shape[0]
assert all(-env.action_space.low == env.action_space.high)
agent_action_limit = env.action_space.high

agent = core.MLPActorCritic(agent_input_dim, agent_output_dim, agent_action_limit)
# agent.load_state_dict(torch.load('weights/agent.pth'))


# simulation
def main():
    run = True
    while run:
        obs, info = env.reset()
        obs = np.concatenate((obs['observation'], obs['desired_goal']))

        total_reward = 0
        for t in range(horizon):
            
            # Simulation
            action = agent.noisy_action(obs, 0.1) # sample agent action
            obs, reward, terminated, truncated, info = env.step(action)  # take action in the environment
            obs = np.concatenate((obs['observation'], obs['desired_goal']))

            total_reward += reward
            if info['is_success']:
                print('success!')

            if terminated or truncated:
                break
    
        print(f"total_reward = {total_reward}")


if __name__ == '__main__':
    main()
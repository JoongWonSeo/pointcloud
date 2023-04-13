# This assumes the gym environment is a GoalEnv as defined by gymnasium_robotics.
import argparse
import numpy as np
import gymnasium as gym
import pointcloud_vision
from robosuite_envs.utils import *
from rl import core

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help='environment ID')
parser.add_argument('--horizon', type=int, default=50, help='horizon')
# parser.add_argument('--sensor', default='default', choices=['default', 'GT', 'PC'], help='sensor')
# parser.add_argument('--obs_encoder', default='default', choices=['default', 'passthru', 'ae'], help='observation encoder')
# parser.add_argument('--goal_encoder', default='PointCloudGTPredictor', help='goal encoder')
args = parser.parse_args()


# global variables
horizon = args.horizon
task = args.env

# setup environment and agent
env = gym.make(task, render_mode='human', max_episode_steps=horizon)

agent_input_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
agent_output_dim = env.action_space.shape[0]
assert all(-env.action_space.low == env.action_space.high)
agent_action_limit = env.action_space.high

# agent = core.MLPActorCritic(agent_input_dim, agent_output_dim, agent_action_limit)


# simulation
run = True
while run:
    obs, info = env.reset()
    obs = np.concatenate((obs['observation'], obs['desired_goal']))

    total_reward = 0
    for t in range(horizon):
        # Simulation
        # action = agent.noisy_action(obs, 0) # sample agent action
        action = np.random.randn(agent_output_dim) # sample random action
        obs, reward, terminated, truncated, info = env.step(action)  # take action in the environment
        obs = np.concatenate((obs['observation'], obs['desired_goal']))

        total_reward += reward
        if info['is_success']:
            print('s', end='')
        
        if env.viewer.is_pressed('g'):
            env.show_frame(env.goal_state, None)

        if terminated or truncated:
            break


        # change state to however we want
        # def change_state(robo_env):
        #     set_obj_pos(robo_env.sim, joint='cube_joint0')
        #     set_robot_pose(robo_env.sim, robo_env.robots[0], np.random.randn(7))
        
        # fake_state = env.render_state(change_state)
        # env.show_frame(fake_state, None)

        # env.show_frame(env.robo_env._get_observations(force_update=True), None)


    print(f"\ntotal_reward = {total_reward}")

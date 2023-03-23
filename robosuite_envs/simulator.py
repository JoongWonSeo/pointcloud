# TODO: move this to root dir and change this to a simple environment demo without any dependencies on rl and pointcloud_vision

# This assumes the gym environment is a GoalEnv as defined by gymnasium_robotics.
import numpy as np
import gymnasium as gym
from pointcloud_vision import PointCloudSensor, PointCloudGTPredictor
from robosuite_envs.utils import *
from rl import core

# global variables
horizon = 1000

# setup environment and agent
# cube_encoder = PointCloudGTPredictor('robot0_eef_pos')

# task = 'RobosuitePickAndPlace-v0'
# task = 'VisionReach-v0'
task = 'VisionLift-v0'
# task = 'VisionPickAndPlace-v0'
# TODO: goal encoder for this needs rerendering!!!! because it is based on the point cloud not the ground truth
env = gym.make(task, render_mode='human', max_episode_steps=horizon)

agent_input_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
agent_output_dim = env.action_space.shape[0]
assert all(-env.action_space.low == env.action_space.high)
agent_action_limit = env.action_space.high

agent = core.MLPActorCritic(agent_input_dim, agent_output_dim, agent_action_limit)
# agent.load_state_dict(torch.load('../rl/weights/agent_succ.pth'))


# simulation
def main():
    run = True
    while run:
        obs, info = env.reset()
        obs = np.concatenate((obs['observation'], obs['desired_goal']))

        total_reward = 0
        for t in range(horizon):
            # Simulation
            action = agent.noisy_action(obs, 0) # sample agent action
            # action = np.random.randn(agent_output_dim) # sample random action
            obs, reward, terminated, truncated, info = env.step(action)  # take action in the environment
            obs = np.concatenate((obs['observation'], obs['desired_goal']))

            total_reward += reward
            if info['is_success']:
                print('s', end='')
            
            if env.viewer.is_pressed('g'):
                env.show_frame(env.episode_goal_state, None)

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


if __name__ == '__main__':
    main()
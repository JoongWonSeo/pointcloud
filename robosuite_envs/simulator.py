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
# task = 'VisionLift-v0'
task = 'VisionPickAndPlace-v0'
# TODO: goal encoder for this needs rerendering!!!! because it is based on the point cloud not the ground truth
env = gym.make(task, render_mode='human', max_episode_steps=horizon)

agent_input_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
agent_output_dim = env.action_space.shape[0]
assert all(-env.action_space.low == env.action_space.high)
agent_action_limit = env.action_space.high

agent = core.MLPActorCritic(agent_input_dim, agent_output_dim, agent_action_limit)


# simulation
def main():
    run = True
    while run:
        obs, info = env.reset()
        obs = np.concatenate((obs['observation'], obs['desired_goal']))

        for t in range(horizon):
            # randomize env
            set_obj_pos(env.robo_env.sim, joint='cube_joint0')
            env.robo_env.robots[0].set_robot_joint_positions(np.random.randn(7))
            # render
            obs = env.robo_env._get_observations(force_update=True)
            env.render_frame(obs, info)



if __name__ == '__main__':
    main()
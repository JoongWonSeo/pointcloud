import argparse
import numpy as np
import torch
import gymnasium as gym
import robosuite_envs
from robosuite_envs.utils import set_obj_pos, random_action
from pointcloud_vision.pc_encoder import PointCloudSensor

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--env', type=str, default='RobosuitePickAndPlace-v0')
parser.add_argument('--horizon', type=int, default=100)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=128)
arg = parser.parse_args()

# global variables
horizon = arg.horizon
runs = arg.runs
total_steps = horizon * runs

env = gym.make(args.env, max_episode_steps=horizon, sensor=PointCloudSensor)


# simulation
step = 0
for r in range(runs):
    env.reset()
    
    for t in range(horizon):        
        set_obj_pos(env.robo_env.sim, joint='cube_joint0')
        #robot.set_robot_joint_positions(np.random.randn(7))
        #robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))

        # Simulation
        #action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        action = random_action(env) # sample random action
        env.step(action)  # take action in the environment

        # convert all torch tensors to numpy arrays
        obs = env.observation.copy()
        # remove all state information
        for k in env.raw_state:
            obs.pop(k)
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                obs[k] = v.cpu().numpy()
        
        ground_truth = env.encoding # proprioception is not included
        np.savez(
            f'{arg.dir}/{step}.npz',
            ground_truth=ground_truth,
            classes=np.array(env.classes, dtype=object),
            **obs
        )

        step += 1
        
        print(('#' * round(step/total_steps * 100)).ljust(100, '-'), end='\r')
print('\ndone')



import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import robosuite_envs
from pointcloud_vision.pc_encoder import PointCloudSensor
from robosuite_envs.utils import set_obj_pos, random_action
from pointcloud_vision.utils import SampleRandomPoints, SampleFurthestPoints

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--env', type=str, default='RobosuitePickAndPlace-v0')
parser.add_argument('--horizon', type=int, default=50)
parser.add_argument('--runs', type=int, default=40)
parser.add_argument('--steps_per_action', type=int, default=5)
parser.add_argument('--actions_per_frame', type=int, default=1)
parser.add_argument('--render', action='store_true')
parser.add_argument('--show_distribution', action='store_true')
arg = parser.parse_args()

# global variables
horizon = arg.horizon
runs = arg.runs
total_steps = horizon * runs

env = gym.make(arg.env, max_episode_steps=horizon, sensor=PointCloudSensor, render_mode='human' if arg.render else None)

# stats
if arg.show_distribution:
    all_points = np.array([]).reshape(0, 6)
    all_gt = np.array([]).reshape(0, 6)
    all_goals = np.array([]).reshape(0, 6)

# simulation
step = 0
for r in range(runs):
    env.reset()

    if arg.show_distribution:
        # save goal
        if env.episode_goal_encoding.shape[0] == 3: # assume it's a point
            x, y, z = env.episode_goal_encoding
            goal = np.array([[x, y, z, 0, 1, 0]])
            all_goals = np.concatenate((all_goals, goal))
    
    for t in range(horizon):        
        # env-defined randomization function (only for non-agent controllable states)
        env.randomize()

        # Simulation
        for i in range(arg.actions_per_frame):
            action = random_action(env) # sample random action
            for j in range(arg.steps_per_action):
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

        if arg.show_distribution:
            pc = np.concatenate((obs['points'], obs['rgb']), axis=1)
            all_points = np.concatenate((all_points, pc))
            
            # save ground truth
            if ground_truth.shape[0] == 3: # assume it's a point
                x, y, z = ground_truth
                gt = np.array([[x, y, z, 1, 0, 0]])
                all_gt = np.concatenate((all_gt, gt))

        step += 1
        
        print(('#' * round(step/total_steps * 100)).ljust(100, '-'), end='\r')
print('\ndone')

if arg.show_distribution:
    print('all points gathered', all_points.shape)
    max_points = 20000
    if all_points.shape[0] + all_gt.shape[0] + all_goals.shape[0] > max_points:
        sampled = SampleRandomPoints(max_points - all_gt.shape[0] - all_goals.shape[0])(all_points)
        all_points = np.concatenate((sampled, all_gt, all_goals))
        print('sampled', all_points.shape)

    points = all_points[:, :3]
    rgb = all_points[:, 3:]
    np.savez(
        f'{arg.dir}/distribution.npz',
        points=points,
        rgb=rgb,
    )
    # rename the file ending to .npz_ignore to prevent it from being loaded by the dataset
    os.rename(f'{arg.dir}/distribution.npz', f'{arg.dir}/distribution.npz_ignore')

    from pc_viewer import plot_pointcloud
    plot_pointcloud({'points': points, 'rgb': rgb})

env.close()
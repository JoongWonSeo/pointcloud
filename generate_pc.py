import array
import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import cv2
from robosuite.utils import camera_utils, transform_utils
from scipy.spatial.transform import Rotation as R
import pandas as pd
import random
from utils import *

# global variables
num_frames=100
camera_w, camera_h = 128, 128

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    render_gpu_device_id=0,
    use_camera_obs=True,
    camera_names=['agentview', 'frontview'],
    camera_widths=camera_w,
    camera_heights=camera_h,
    camera_depths=True,
    horizon=num_frames,
)

robot = env.robots[0]
# print(f"limits = {robot.action_limits}\naction_dim = {robot.action_dim}\nDoF = {robot.dof}")

# create camera mover
camera_l = camera_utils.CameraMover(env, camera='frontview')
camera_r = camera_utils.CameraMover(env, camera='agentview')
camera_l.set_camera_pose([0, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))
camera_r.set_camera_pose([0, 1.2, 1.8], transform_utils.axisangle2quat([-0.817, 0, 0]))
camera_r.rotate_camera(None, (0, 0, 1), 180)

# simulation
def main():

    for t in range(num_frames):
        
        set_obj_pos(env.sim, joint='cube_joint0')
        robot.set_robot_joint_positions(np.random.randn(7))
        #robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))


        # Simulation
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        obs, reward, done, info = env.step(action)  # take action in the environment

        # Observation
        depth_map_l = camera_utils.get_real_depth_map(env.sim, obs['frontview_depth'])
        depth_map_r = camera_utils.get_real_depth_map(env.sim, obs['agentview_depth'])

        rgb_l = obs['frontview_image'] / 255
        rgb_r = obs['agentview_image'] / 255

        # combine pointclouds
        pc_l, rgb_l = to_pointcloud(env.sim, rgb_l, depth_map_l, 'frontview')
        pc_r, rgb_r = to_pointcloud(env.sim, rgb_r, depth_map_r, 'agentview')
        pc = np.concatenate((pc_l, pc_r), axis=0)
        rgb = np.concatenate((rgb_l, rgb_r), axis=0)

        # filter out points outside of bounding box
        bbox = np.array([[-0.5, 0.5], [-0.5, 0.5], [0, 1.5]])
        pc, rgb = filter_pointcloud(pc, rgb, bbox)

        # random sampling to fixed number of points
        n = 10000
        idx = np.random.choice(pc.shape[0], n, replace=False)
        pc = pc[idx, :]
        rgb = rgb[idx, :]

        np.savez(f'input/{t}.npz', points=pc, rgb=rgb)
        
        print(f"number of points = {pc.shape[0]}")

    

if __name__ == '__main__':
    main()
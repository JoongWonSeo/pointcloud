import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import cv2
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
import pandas as pd
import random
from utils import *

# global variables
horizon=10000
camera_w, camera_h = 512, 512

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    render_gpu_device_id=0,
    use_camera_obs=True,
    camera_widths=camera_w,
    camera_heights=camera_h,
    camera_depths=True,
    camera_segmentations='instance',
    horizon=horizon,
    # renderer='igibson'
)

robot = env.robots[0]
print(f"limits = {robot.action_limits}\naction_dim = {robot.action_dim}\nDoF = {robot.dof}")

# create camera mover
camera = camera_utils.CameraMover(env, camera='agentview')


# simulation
def main():
    ui = UI('RGBD', camera)

    for t in range(horizon):
        # UI interaction
        if not ui.update():
            break
        
        # set cube position        
        if ui.is_pressed('r'):
            set_obj_pos(env.sim, joint='cube_joint0')
            #robot.set_robot_joint_positions(np.random.randn(7))
            robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))


        # Simulation
        #action = random_action(env) # sample random action
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        obs, reward, done, info = env.step(action)  # take action in the environment

        # Observation
        depth_map = camera_utils.get_real_depth_map(env.sim, obs['agentview_depth'])
        
        if ui.is_pressed('p'):
            save_pointcloud(env.sim, depth_map, camera='agentview', w=camera_w, h=camera_h)

        rgb, d = obs['agentview_image'] / 255, normalize(depth_map)
        s = normalize(obs['agentview_segmentation_instance'])
        rgbd = np.flip(np.hstack((rgb, np.dstack((d, d, d)), np.dstack((s, s, s)))), axis=0)

        ui.show(rgbd)
    
    ui.close()


if __name__ == '__main__':
    main()
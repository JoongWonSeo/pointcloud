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
horizon=10000
camera_w, camera_h = 256, 256 #512, 512

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
    #camera_segmentations='instance',
    horizon=horizon,
)

robot = env.robots[0]
print(f"limits = {robot.action_limits}\naction_dim = {robot.action_dim}\nDoF = {robot.dof}")

# create camera mover
camera = camera_utils.CameraMover(env, camera='agentview')
camera_r = camera_utils.CameraMover(env, camera='frontview')
# camera_r.set_camera_pose([0, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))
# camera.set_camera_pose([0, 1.2, 1.8], transform_utils.axisangle2quat([-0.817, 0, 0]))
# camera.rotate_camera(None, (0, 0, 1), 180)
camera.set_camera_pose([-0.2, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))
camera_r.set_camera_pose([0.2, -1.2, 1.8], transform_utils.axisangle2quat([1, 0, 0]))


# simulation
def main():
    ui = UI('RGBD', camera)

    for t in range(horizon):
        # UI interaction
        if not ui.update():
            break

        # get camera pose
        if ui.is_pressed('c'):
            #camera.set_camera_pose([0.9, -0.2, 1.8])
            camera_pose = camera.get_camera_pose()
            angles = transform_utils.quat2axisangle(camera_pose[1])
            print(f"camera_pose = {camera_pose}")
            print(f"angles = {angles}")
        
        # set cube position        
        if ui.is_pressed('r'):
            set_obj_pos(env.sim, joint='cube_joint0')
            robot.set_robot_joint_positions(np.random.randn(7))
            #robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))

        # Simulation
        #action = random_action(env) # sample random action
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        obs, reward, done, info = env.step(action)  # take action in the environment

        # Observation
        depth_map = camera_utils.get_real_depth_map(env.sim, obs['agentview_depth'])

        rgb, d = obs['agentview_image'] / 255, normalize(depth_map)

        #DEBUG get cube position
        cp = env.sim.data.get_joint_qpos('cube_joint0')[:3]
        w2c = camera_utils.get_camera_transform_matrix(env.sim, 'agentview', camera_h, camera_w)
        # cp to homogeneous coordinates
        cp = np.append(cp, 1)
        ci = w2c @ cp
        # ci to pixel coordinates
        ci = ci[:2] / ci[2]
        # ci to image coordinates
        ci = np.round(ci).astype(int)
        x, y = ci[0], ci[1]
        _y = camera_h - y
        #print(f"x, y = {x}, {y}")
        # draw cube position
        rgb[_y-2:_y+2, x-2:x+2] = [1, 1, 0]

        # find original coordinate
        c2w = np.linalg.inv(w2c)
        z = depth_map[_y, x]
        ci = np.array([x*z, y*z, z, 1], dtype=np.float32)
        cp_reconst = c2w @ ci
        cp_reconst = cp_reconst[:3] / cp_reconst[3]
        cp_reconst = np.append(cp_reconst, 1)
        print(f"cp = {cp}\ncp_reconst = {cp_reconst}")
        print(f"diff = {cp - cp_reconst}")



        rgbd = np.flip(np.hstack((rgb, np.dstack((d, d, d)))), axis=0)

        
        depth_map_r = camera_utils.get_real_depth_map(env.sim, obs['frontview_depth'])

        rgb_r, d_r = obs['frontview_image'] / 255, normalize(depth_map_r)
        rgbd_r = np.flip(np.hstack((rgb_r, np.dstack((d_r, d_r, d_r)))), axis=0)


        if ui.is_pressed('p'):
            save_pointcloud(env.sim, rgb, depth_map, camera='agentview', file='output/pc.npz')

        ui.show(np.vstack((rgbd, rgbd_r)))
    
    ui.close()


if __name__ == '__main__':
    main()
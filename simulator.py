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
import torch
from models.pn_autoencoder import PNAutoencoder

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
    camera_names=['agentview'],
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
camera.set_camera_pose([-0.2, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))



# import PC AE
device = 'cuda'
ae = PNAutoencoder(2048, 6)
ae.load_state_dict(torch.load('weights/PC_AE.pth'))
ae = ae.to(device)
ae.eval()


# simulation
def main():
    ui = UI('RGBD', camera)

    for t in range(horizon):
        # UI interaction
        if not ui.update():
            break

        # get camera pose
        if ui.is_pressed('c'):
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
        action = random_action(env) * 10 # sample random action
        # action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        obs, reward, done, info = env.step(action)  # take action in the environment

        # Observation
        rgb = obs['agentview_image'] / 255

        # Render autoencoder's generated PC
        # capture the point cloud
        depth_map = camera_utils.get_real_depth_map(env.sim, obs['agentview_depth'])
        orig, orig_rgb = to_pointcloud(env.sim, rgb, depth_map, 'agentview')
        # orig, orig_rgb = torch.Tensor(orig), torch.Tensor(orig_rgb)
        # orig = torch.stack([orig, orig_rgb], dim=1)
        orig = np.concatenate([orig, orig_rgb], axis=1)

        # sample random 2048 points TODO: more uniform dense sampling
        orig = orig[np.random.choice(orig.shape[0], 2048, replace=False), :]

        # normalize the points
        bbox = np.array((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5))
        min = bbox[0:6:2]
        max = bbox[1:6:2]
        orig[:, :3] = (orig[:, :3] - min) / (max - min)

        # run the autoencoder
        orig = torch.Tensor(orig).to(device)
        orig = orig.reshape((1, ae.out_points, ae.dim_per_point))
        embedding = ae.encoder(orig)
        # embedding = torch.randn(ae.encoder.out_channels).to(device)
        print(embedding)
        pred = ae.decoder(embedding).reshape((ae.out_points, ae.dim_per_point)).detach().cpu().numpy()

        # unnormalize the points
        pred[:, :3] = min + pred[:, :3] * (max - min)

        w2c = camera_utils.get_camera_transform_matrix(env.sim, 'agentview', camera_h, camera_w)
        
        # DEBUG make image more blue
        rgb[:, :, 0:2] = rgb[:, :, 0:2] * 0.5
        render(pred[:, :3], pred[:, 3:], rgb, w2c, camera_h, camera_w)


        # convert to CV2 format (flip along y axis and from RGB to BGR)
        rgb = np.flip(rgb, axis=0)
        rgb = rgb[:, :, [2, 1, 0]]

        ui.show(rgb)
    
    ui.close()


if __name__ == '__main__':
    main()
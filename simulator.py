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
from pytorch3d.ops import sample_farthest_points

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
    camera_names=['agentview', 'frontview', 'birdview', 'sideview'],
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

# pc observer
sens_l = camera_utils.CameraMover(env, camera='frontview')
sens_r = camera_utils.CameraMover(env, camera='birdview')
sens_t = camera_utils.CameraMover(env, camera='sideview')
sens_l.set_camera_pose([0, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))
sens_r.set_camera_pose([0, 1.2, 1.8], transform_utils.axisangle2quat([-0.817, 0, 0]))
sens_r.rotate_camera(None, (0, 0, 1), 180)
sens_t.set_camera_pose([0, 0, 1.7], transform_utils.axisangle2quat([0, 0, 0]))



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
        camera_image = obs['agentview_image'] / 255

        # Sensor images
        sens_l_image = obs['frontview_image'] / 255
        sens_r_image = obs['birdview_image'] / 255
        sens_t_image = obs['sideview_image'] / 255
        sens_l_depth = camera_utils.get_real_depth_map(env.sim, obs['frontview_depth'])
        sens_r_depth = camera_utils.get_real_depth_map(env.sim, obs['birdview_depth'])
        sens_t_depth = camera_utils.get_real_depth_map(env.sim, obs['sideview_depth'])


        # RGBD to point cloud
        pc_l, pc_l_rgb = to_pointcloud(env.sim, sens_l_image, sens_l_depth, 'frontview')
        pc_r, pc_r_rgb = to_pointcloud(env.sim, sens_r_image, sens_r_depth, 'birdview')
        pc_t, pc_t_rgb = to_pointcloud(env.sim, sens_t_image, sens_t_depth, 'sideview')
        orig = np.concatenate([pc_l, pc_r, pc_t], axis=0)
        orig_rgb = np.concatenate([pc_l_rgb, pc_r_rgb, pc_t_rgb], axis=0)

        # filter, sample and normalize
        bbox = np.array([[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]])
        orig, orig_rgb = filter_pointcloud(orig, orig_rgb, bbox)
        orig = np.concatenate([orig, orig_rgb], axis=1)

        # sample random 2048 points TODO: more uniform dense sampling
        # orig = orig[np.random.choice(orig.shape[0], 2048, replace=False), :]
        orig = torch.Tensor(orig).to(device)
        orig = orig.reshape((1, -1, ae.dim_per_point))
        orig, _ = sample_farthest_points(orig, K=2048)

        # normalize the points
        bbox = torch.Tensor((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5)).to(device)
        min = bbox[0:6:2]
        max = bbox[1:6:2]
        orig[:, :, :3] = (orig[:, :, :3] - min) / (max - min)

        # run the autoencoder
        pred = ae(orig).reshape((ae.out_points, ae.dim_per_point)).detach()
        #pred = orig # DEBUG use original point cloud

        # unnormalize the points
        pred[:, :3] = min + pred[:, :3] * (max - min)
        pred = pred.cpu().numpy()

        # render to camera
        w2c = camera_utils.get_camera_transform_matrix(env.sim, 'agentview', camera_h, camera_w)
        # DEBUG make image more blue
        camera_image[:, :, 0:2] = camera_image[:, :, 0:2] * 0.5
        # render(pred[:, :3], pred[:, 3:], camera_image, w2c, camera_h, camera_w)
        render(pred[:, :3], pred[:, [5, 3, 4]], camera_image, w2c, camera_h, camera_w)


        # convert to CV2 format (flip along y axis and from RGB to BGR)
        camera_image = np.flip(camera_image, axis=0)
        camera_image = camera_image[:, :, [2, 1, 0]]

        ui.show(camera_image)
    
    ui.close()


if __name__ == '__main__':
    main()
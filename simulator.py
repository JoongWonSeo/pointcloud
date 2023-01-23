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
import rl_core as core
# global variables
horizon = 10000
camera_w, camera_h = 256, 256 #512, 512

# create environment instance
# env = suite.make(
#     env_name="Lift", # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     has_renderer=False,
#     has_offscreen_renderer=True,
#     render_gpu_device_id=0,
#     use_camera_obs=True,
#     camera_names=['agentview', 'frontview', 'birdview', 'sideview'],
#     camera_widths=camera_w,
#     camera_heights=camera_h,
#     camera_depths=True,
#     #camera_segmentations='instance',
#     horizon=horizon,
# )

def make_env():
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=False, # sparse reward
    )

    # wrap the observation space for compatibility
    def get_observation(obs=None):
        # object position concatenated with robot joint angles
        if obs is None:
            obs = env._get_observations()
        return np.concatenate((obs['object-state'], obs['robot0_proprio-state']))
    env.get_observation = get_observation

    # create a initial state to task goal mapper, specific to the task
    def get_task_goal(obs=None):
        if obs is None:
            obs = env.get_observation()
        goal = obs[:3]
        goal[2] += 0.05 # lift the object by 5cm (in the lift task, it's defined as 4cm above table height)
        return goal
    env.get_task_goal = get_task_goal

    # create a state to goal mapper for HER, such that the input state safisfies the returned goal
    def as_goal(obs=None):
        if obs is None:
            obs = env.get_observation()
        return obs[:3] # object position
    env.as_goal = as_goal

    # create a state-goal to reward function (sparse)
    def get_reward(obs, goal):
        # as long as the object is close enough to the goal, the reward is 1
        return 0 if np.linalg.norm(obs[:3] - goal) < 0.05 else -1
    env.get_reward = get_reward

    # wrap the step function to return the wrapped observation
    env._step = env.step
    def step(action):
        obs, reward, done, info = env._step(action)
        return get_observation(obs), reward, done, info
    env.step = step

    # wrap the reset function to return the wrapped observation
    env._reset = env.reset
    def reset():
        return get_observation(env._reset())
    env.reset = reset

    return env
env = make_env()

robot = env.robots[0]
print(f"limits = {robot.action_limits}\naction_dim = {robot.action_dim}\nDoF = {robot.dof}")


# pc observer
# sens_l = camera_utils.CameraMover(env, camera='frontview')
# sens_r = camera_utils.CameraMover(env, camera='birdview')
# sens_t = camera_utils.CameraMover(env, camera='sideview')
# sens_l.set_camera_pose([0, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))
# sens_r.set_camera_pose([0, 1.2, 1.8], transform_utils.axisangle2quat([-0.817, 0, 0]))
# sens_r.rotate_camera(None, (0, 0, 1), 180)
# sens_t.set_camera_pose([0, 0, 1.7], transform_utils.axisangle2quat([0, 0, 0]))



# import PC AE
device = 'cuda'
# ae = PNAutoencoder(2048, 6)
# ae.load_state_dict(torch.load('weights/PC_AE.pth'))
# ae = ae.to(device)
# ae.eval()

# create agent
obs_dim = env.get_observation().shape[0]
goal_dim = env.get_task_goal().shape[0]
og_dim = obs_dim + goal_dim
act_dim = env.action_dim
act_limit = env.action_spec[1][0]

agent = core.MLPActorCritic(og_dim, act_dim, act_limit)
agent.load_state_dict(torch.load('weights/agent.pth'))

def get_action(o, noise_scale):
    a = agent.act(torch.as_tensor(o, dtype=torch.float32))
    a += noise_scale * np.random.randn(act_dim)
    return np.clip(a, -act_limit, act_limit)


# simulation
def main():
    ui = UI('RGBD', None)

    run = True
    while run:
        obs = env.reset()
        # set_obj_pos(env.sim, joint='cube_joint0')
        # obs = env.get_observation()
        goal = env.get_task_goal()
        og = np.concatenate((obs, goal))

        total_reward = 0

        # create camera mover
        ui.camera = camera = camera_utils.CameraMover(env, camera='agentview')
        # move the camera back a bit
        camera.move_camera((0,0,1), 0.5) # z axis (forward-backward)

        for t in range(horizon):
            # UI interaction
            if not ui.update():
                run = False
                break
            
            # set cube position        
            if ui.is_pressed('r'):
                break
                # set_obj_pos(env.sim, joint='cube_joint0')
                # robot.set_robot_joint_positions(np.random.randn(7))
                #robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))

            # Simulation
            action = get_action(og, 0) # sample agent action
            # action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            obs, reward, done, info = env.step(action)  # take action in the environment
            og = np.concatenate((obs, goal))
            reward = env.get_reward(obs, goal)
            total_reward += reward

            # Observation
            camera_image = env._get_observations()['agentview_image'] / 255

            # # Sensor images
            # sens_l_image = obs['frontview_image'] / 255
            # sens_r_image = obs['birdview_image'] / 255
            # sens_t_image = obs['sideview_image'] / 255
            # sens_l_depth = camera_utils.get_real_depth_map(env.sim, obs['frontview_depth'])
            # sens_r_depth = camera_utils.get_real_depth_map(env.sim, obs['birdview_depth'])
            # sens_t_depth = camera_utils.get_real_depth_map(env.sim, obs['sideview_depth'])


            # # RGBD to point cloud
            # pc_l, pc_l_rgb = to_pointcloud(env.sim, sens_l_image, sens_l_depth, 'frontview')
            # pc_r, pc_r_rgb = to_pointcloud(env.sim, sens_r_image, sens_r_depth, 'birdview')
            # pc_t, pc_t_rgb = to_pointcloud(env.sim, sens_t_image, sens_t_depth, 'sideview')
            # orig = np.concatenate([pc_l, pc_r, pc_t], axis=0)
            # orig_rgb = np.concatenate([pc_l_rgb, pc_r_rgb, pc_t_rgb], axis=0)

            # # filter, sample and normalize
            # bbox = np.array([[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]])
            # orig, orig_rgb = filter_pointcloud(orig, orig_rgb, bbox)
            # orig = np.concatenate([orig, orig_rgb], axis=1)

            # # sample random 2048 points TODO: more uniform dense sampling
            # orig = torch.Tensor(orig).to(device)
            # orig = orig.reshape((1, -1, ae.dim_per_point))
            # # orig = orig[:, np.random.choice(orig.shape[1], 2048, replace=False), :]
            # orig, _ = sample_farthest_points(orig, K=2048)

            # # normalize the points
            # bbox = torch.Tensor((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5)).to(device)
            # min = bbox[0:6:2]
            # max = bbox[1:6:2]
            # orig[:, :, :3] = (orig[:, :, :3] - min) / (max - min)

            # # run the autoencoder
            # pred = ae(orig).reshape((ae.out_points, ae.dim_per_point)).detach()
            # # pred = orig.reshape((ae.out_points, ae.dim_per_point)) # DEBUG use original point cloud

            # # unnormalize the points
            # pred[:, :3] = min + pred[:, :3] * (max - min)
            # pred = pred.cpu().numpy()

            # # render to camera
            # w2c = camera_utils.get_camera_transform_matrix(env.sim, 'agentview', camera_h, camera_w)
            # # DEBUG make image more blue
            # camera_image[:, :, 0:2] = camera_image[:, :, 0:2] * 0.5
            # # render(pred[:, :3], pred[:, 3:], camera_image, w2c, camera_h, camera_w)
            # render(pred[:, :3], pred[:, [5, 3, 4]], camera_image, w2c, camera_h, camera_w)


            # convert to CV2 format (flip along y axis and from RGB to BGR)
            camera_image = np.flip(camera_image, axis=0)
            camera_image = camera_image[:, :, [2, 1, 0]]

            ui.show(camera_image)
    
        print(f"total_reward = {total_reward}")

    ui.close()


if __name__ == '__main__':
    main()
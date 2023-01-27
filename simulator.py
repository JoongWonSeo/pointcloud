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
from simulation_adapter import MultiGoalEnvironment, make_multigoal_lift

# global variables
horizon = 10000
camera_w, camera_h = 256, 256 #512, 512

# setup environment and agent
env = make_multigoal_lift(horizon)
agent = core.MLPActorCritic(env.obs_dim, env.action_dim, env.act_limit)
agent.load_state_dict(torch.load('weights/agent.pth'))


# simulation
def main():
    ui = UI('RGBD', None)

    run = True
    while run:
        obs = env.reset()
        goal = env.desired_goal(obs)

        total_reward = 0

        # create camera mover
        ui.camera = camera_utils.CameraMover(env, camera='agentview')
        ui.camera.move_camera((0,0,1), 0.5) # move the camera back a bit

        for t in range(horizon):
            # UI interaction
            if not ui.update():
                run = False
                break
            
            # restart
            if ui.is_pressed('r'):
                break

            # Simulation
            action = agent.noisy_action(obs, 0.1) # sample agent action
            obs, reward, done, info = env.step(action)  # take action in the environment
            total_reward += reward
            if env.check_success(env.achieved_goal(obs), goal, info):
                print('success!')
                # break
            if done:
                print('done!')
                # sleep
                import time
                time.sleep(1)
                break

            # Render
            camera_image = env.get_camera_image('agentview')
            #DEBUG: visualize goal pos and gripper pos
            goal_pos = obs[env.only_obs_dim:env.only_obs_dim+3]
            gripper_pos = env._get_observations()['robot0_eef_pos']
            
            # to batch of points
            points = np.stack((goal_pos, gripper_pos), axis=0)
            rgb = np.array([[1, 0, 0], [0, 0, 1]])
            w2c = camera_utils.get_camera_transform_matrix(env.sim, 'agentview', camera_h, camera_w)
            render(points, rgb, camera_image, w2c, camera_h, camera_w)
            ui.show(to_cv2_img(camera_image))
    
        print(f"total_reward = {total_reward}")

    ui.close()


if __name__ == '__main__':
    main()
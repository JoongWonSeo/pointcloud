import numpy as np
import robosuite as suite
from robosuite.utils import camera_utils, transform_utils
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--frames', type=int, default=100)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=128)
arg = parser.parse_args()

# global variables
num_frames=arg.frames
camera_w, camera_h = arg.width, arg.height

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    render_gpu_device_id=0,
    use_camera_obs=True,
    camera_names=['agentview', 'frontview', 'birdview'],
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
camera_t = camera_utils.CameraMover(env, camera='birdview')
camera_l.set_camera_pose([0, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))
camera_r.set_camera_pose([0, 1.2, 1.8], transform_utils.axisangle2quat([-0.817, 0, 0]))
camera_r.rotate_camera(None, (0, 0, 1), 180)
camera_t.set_camera_pose([0, 0, 1.7], transform_utils.axisangle2quat([0, 0, 0]))

# simulation
def main():

    for t in range(num_frames):
        
        set_obj_pos(env.sim, joint='cube_joint0')
        #robot.set_robot_joint_positions(np.random.randn(7))
        robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))


        # Simulation
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        obs, _, _, _ = env.step(action)  # take action in the environment

        # Observation
        depth_map_l = camera_utils.get_real_depth_map(env.sim, obs['frontview_depth'])
        depth_map_r = camera_utils.get_real_depth_map(env.sim, obs['agentview_depth'])
        depth_map_t = camera_utils.get_real_depth_map(env.sim, obs['birdview_depth'])

        # normalize rgb to [0, 1]
        rgb_l = obs['frontview_image'] / 255
        rgb_r = obs['agentview_image'] / 255
        rgb_t = obs['birdview_image'] / 255

        # combine pointclouds
        pc_l, rgb_l = to_pointcloud(env.sim, rgb_l, depth_map_l, 'frontview')
        pc_r, rgb_r = to_pointcloud(env.sim, rgb_r, depth_map_r, 'agentview')
        pc_t, rgb_t = to_pointcloud(env.sim, rgb_t, depth_map_t, 'birdview')
        pc = np.concatenate((pc_l, pc_r, pc_t), axis=0)
        rgb = np.concatenate((rgb_l, rgb_r, rgb_t), axis=0)

        # filter out points outside of bounding box
        bbox = np.array([[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]])
        # bbox = np.array([[-0.4, 0.4], [-0.4, 0.4], [0.8, 1.5]])
        pc, rgb = filter_pointcloud(pc, rgb, bbox)

        # # random sampling to fixed number of points
        # n = 10000
        # idx = np.random.choice(pc.shape[0], n, replace=False)
        # pc = pc[idx, :]
        # rgb = rgb[idx, :]

        np.savez(f'input/{t}.npz', points=pc, features=rgb, boundingbox=bbox)
        
        print(f"number of points = {pc.shape[0]}")

    

if __name__ == '__main__':
    main()
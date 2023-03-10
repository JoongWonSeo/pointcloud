import cfg
import numpy as np
import robosuite as suite
from robosuite.utils import camera_utils, transform_utils
import argparse
from torchvision.transforms import Compose
from sim.utils import *
from vision.utils import SampleRandomPoints, SampleFurthestPoints, FilterClasses, FilterBBox, Normalize

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--horizon', type=int, default=100)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=128)
arg = parser.parse_args()

# global variables
horizon = arg.horizon
runs = arg.runs
total_steps = horizon * runs
camera_w, camera_h = arg.width, arg.height
cameras = list(cfg.camera_poses.keys())
num_classes = len(cfg.classes)

# create environment instance
env = suite.make(
    env_name=cfg.env, # try with other tasks like "Stack" and "Door"
    robots=cfg.robot,  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    render_gpu_device_id=0,
    use_camera_obs=True,
    camera_names=cameras,
    camera_widths=camera_w,
    camera_heights=camera_h,
    camera_depths=True,
    camera_segmentations='class',
    horizon=horizon,
)

robot = env.robots[0]


# define transform to apply to pointcloud and ground truth state
bbox = np.array(cfg.bbox)
transform = cfg.pc_preprocessor()
gt_transform = cfg.gt_preprocessor()

# simulation
step = 0
for r in range(runs):
    env.reset()

    # setup cameras (it's important to first create the camera mover objects and then set the camera poses, because the environment is reset every time a mover is created)
    movers = {cam: camera_utils.CameraMover(env, camera=cam) for cam in cameras}
    for camera_name, camera_pose in cfg.camera_poses.items():
        movers[camera_name].set_camera_pose(np.array(camera_pose[0]), np.array(camera_pose[1]))
    
    for t in range(horizon):        
        set_obj_pos(env.sim, joint='cube_joint0')
        #robot.set_robot_joint_positions(np.random.randn(7))
        #robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))

        # Simulation
        #action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        action = random_action(env) # sample random action
        obs, _, _, _ = env.step(action)  # take action in the environment

        pc, feats = multiview_pointcloud(env.sim, obs, cameras, transform, ['rgb', 'segmentation'], num_classes)
        ground_truth = np.concatenate([t(obs[key]) for key, t in gt_transform.items()], axis=0)
        np.savez(
            f'{arg.dir}/{step}.npz',
            points=pc,
            **feats,
            ground_truth=ground_truth,
            boundingbox=bbox,
            classes=np.array(cfg.classes, dtype=object)
        )

        step += 1
        
        print(('#' * round(step/total_steps * 100)).ljust(100, '-'), end='\r')
print('\ndone')



import cfg
import numpy as np
import robosuite as suite
from robosuite.utils import camera_utils, transform_utils
import argparse
from torchvision.transforms import Compose
from sim.utils import *
from vision.utils import SampleRandomPoints, SampleFurthestPoints, FilterClasses, FilterBBox, Normalize

parser = argparse.ArgumentParser()
parser.add_argument('--horizon', type=int, default=100)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=128)
parser.add_argument('--dir', default='input')
arg = parser.parse_args()

# global variables
horizon = arg.horizon
runs = arg.runs
total_steps = horizon * runs
camera_w, camera_h = arg.width, arg.height
cameras = ['frontview', 'agentview', 'birdview']
num_classes = len(cfg.classes)

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
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


# define transform to apply to pointcloud
bbox = np.array(cfg.bbox)
transform = Compose([
    # FilterClasses(whitelist=[0, 1], seg_dim=6), # only keep table and cube
    FilterBBox(bbox),
    # SampleRandomPoints(2048),
    SampleFurthestPoints(2048),
    Normalize(bbox)
])

# simulation
step = 0
for r in range(runs):
    env.reset()

    # create camera mover
    camera_l = camera_utils.CameraMover(env, camera=cameras[0])
    camera_r = camera_utils.CameraMover(env, camera=cameras[1])
    camera_t = camera_utils.CameraMover(env, camera=cameras[2])
    camera_l.set_camera_pose([0, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0]))
    camera_r.set_camera_pose([0, 1.2, 1.8], transform_utils.axisangle2quat([-0.817, 0, 0]))
    camera_r.rotate_camera(None, (0, 0, 1), 180)
    camera_t.set_camera_pose([0, 0, 1.7], transform_utils.axisangle2quat([0, 0, 0]))
    
    for t in range(horizon):        
        set_obj_pos(env.sim, joint='cube_joint0')
        #robot.set_robot_joint_positions(np.random.randn(7))
        #robot.set_robot_joint_positions(np.array([-1, 0, 0, 0, 0, 0, 0]))

        # Simulation
        #action = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        action = random_action(env) # sample random action
        obs, _, _, _ = env.step(action)  # take action in the environment

        pc, feats = multiview_pointcloud(env.sim, obs, cameras, transform, ['rgb', 'segmentation'], num_classes)
        np.savez(f'{arg.dir}/{step}.npz', points=pc, **feats, boundingbox=bbox, classes=np.array(cfg.classes, dtype=object))

        step += 1
        
        print(('#' * round(step/total_steps * 100)).ljust(100, '-'), end='\r')
print('\ndone')



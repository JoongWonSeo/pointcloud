import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import cv2
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
import pandas as pd
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer


def pprint_dict(d):
    import json
    print(json.dumps(d, sort_keys=True, indent=4, default=str))

mouse_x = mouse_y = 0
mouse_clicked = False
def mouse_callback(event,x,y,flags,param):
    global mouse_x, mouse_y, mouse_clicked
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True
    if event == cv2.EVENT_LBUTTONUP:
        mouse_clicked = False


def normalize(array):
    '''given np array, stretch its elements to 0 and 1 range'''
    min, max = np.min(array), np.max(array)
    return (array - min) / (max - min)


def pixel_to_world(pixels, depth_map, camera_to_world_transform):
    """
    Helper function to take a batch of pixel locations and the corresponding depth image
    and transform these points from the camera frame to the world frame.

    Args:
        pixels (np.array): N pixel coordinates of shape [N, 2]
        depth_map (np.array): depth image of shape [H, W]
        camera_to_world_transform (np.array): 4x4 Tensor to go from pixel coordinates to world
            coordinates.

    Return:
        points (np.array): N 3D points in robot frame of shape [N, 3]
    """

    # sample from the depth map using the pixel locations
    z = np.array([depth_map[y, x, 0] for x, y in pixels])
    x, y = pixels.T

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    homogenous = np.vstack((x*z, y*z, z, np.ones_like(z)))

    # batch matrix multiplication of 4 x 4 matrix and 4 x N vectors to do camera to robot frame transform
    points = camera_to_world_transform @ homogenous
    return points[:3, ...].T


def policy_action(env):
    return np.random.randn(env.robots[0].dof)


horizon=10000
camera_w, camera_h = 512, 512

# create environment instance
# env = suite.make(
#     env_name="Lift", # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     has_renderer=False,
#     has_offscreen_renderer=True,
#     render_gpu_device_id=0,
#     use_camera_obs=True,
#     camera_depths=True,
#     camera_widths=camera_w,
#     camera_heights=camera_h,
#     horizon=horizon,
#     # renderer='igibson'
# )

# create world and env
world = MujocoWorldBase()
mujoco_robot = Panda()
gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)

# TODO: register this "TASK" (as a class?) to ENV and load it as env


# create camera mover
camera = camera_utils.CameraMover(env, camera='agentview')
# camera.move_camera((0,0,1), 1)
# (x, y, z), quat = camera.get_camera_pose()


# reset the environment
#env.reset()

cv2.namedWindow('RGBD')
cv2.setMouseCallback('RGBD', mouse_callback)

for t in range(horizon):
    pressed = cv2.waitKey(10)
    if pressed == 27: #ESC key
        break
    if pressed == ord('w'):
        camera.move_camera((0,0,1), -0.05)
    if pressed == ord('s'):
        camera.move_camera((0,0,1), 0.05)
    if pressed == ord('a'):
        camera.move_camera((1,0,0), -0.05)
    if pressed == ord('d'):
        camera.move_camera((1,0,0), 0.05)
    if pressed == ord('2'):
        camera.move_camera((0,1,0), 0.05)
    if pressed == ord('z'):
        camera.move_camera((0,1,0), -0.05)
    if pressed == ord('w'):
        camera.move_camera((0,0,1), -0.05)
    

    if mouse_clicked:
        mx_prev, my_prev = mx, my

    mx, my = mouse_x, mouse_y

    if mouse_clicked:
        camera.rotate_camera(point=None, axis=(0, 1, 0), angle=(mx-mx_prev)/10)
        camera.rotate_camera(point=None, axis=(1, 0, 0), angle=(my-my_prev)/10)


    action = policy_action(env) # sample random action

    obs, reward, done, info = env.step(action)  # take action in the environment

    depth_map = camera_utils.get_real_depth_map(env.sim, obs['agentview_depth'])
    
    if pressed == ord('p'):
        all_pixels = np.array([[x, y] for x in range(camera_w) for y in range(camera_h)])
        #all_pixels = np.array([camera_w/2, camera_h/2])
        trans_pix_to_world = np.linalg.inv(camera_utils.get_camera_transform_matrix(env.sim, 'agentview', camera_h, camera_w))
        point_cloud = pixel_to_world(all_pixels, depth_map, trans_pix_to_world)
        df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])
        df.to_csv('point_cloud.csv')

    rgb, d = np.flip(obs['agentview_image'] / 255, axis=0), np.flip(normalize(depth_map), axis=0)
    rgbd = np.hstack((rgb, np.dstack((d, d, d))))

    cv2.imshow('RGBD', rgbd)
 
cv2.destroyAllWindows()
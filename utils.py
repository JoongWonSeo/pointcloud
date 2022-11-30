import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
import cv2
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T
import pandas as pd
import random

def pprint_dict(d):
    import json
    print(json.dumps(d, sort_keys=True, indent=4, default=str))


def normalize(array):
    '''given np array, shift and stretch its elements to 0 and 1 range'''
    min, max = np.min(array), np.max(array)
    return (array - min) / (max - min)



def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def get_camera_extrinsic_matrix(sim, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    return R


def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv

def get_camera_transform_matrix(sim, camera_name, camera_height, camera_width):
    """
    Camera transform matrix to project from world coordinates to pixel coordinates.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
    """
    R = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    K = get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
    )
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    return K_exp @ pose_inv(R)




def pixel_to_world(pixels, depth_map, camera_to_world_transform):
    """
    Helper function to take a batch of pixel locations and the corresponding depth image
    and transform these points from the camera coordinate system to the world coords.

    Args:
        pixels (np.array): N pixel coordinates of shape [N, 2]
        depth_map (np.array): depth image of shape [H, W, 1]
        camera_to_world_transform (np.array): 4x4 matrix to go from pixel coordinates to world
            coordinates.

    Return:
        points (np.array): N 3D points in robot frame of shape [N, 3]
    """

    # sample from the depth map using the pixel locations
    z = np.array([depth_map[-y, x, 0] for x, y in pixels])
    x, y = pixels.T

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    homogenous = np.vstack((x*z, y*z, z, np.ones_like(z)))
    #homogenous = np.vstack((y, x, z, np.ones_like(z)))

    # batch matrix multiplication of 4 x 4 matrix and 4 x N vectors to do camera to robot frame transform
    points = camera_to_world_transform @ homogenous
    # points = points[:3] / points[3]
    points = points[:3]
    return points.T

def pixel_to_feature(pixels, feature_map):
    """
    Helper function to take a batch of pixel locations and the corresponding feature map (image)
    and return the feature vector for each of the pixel location.

    Args:
        pixels (np.array): N pixel coordinates of shape [N, 2]
        feature_map (np.array): feature image of shape [H, W, C]

    Return:
        points (np.array): N 3D points in robot frame of shape [N, C]
    """

    # sample from the feature map using the pixel locations
    features = np.array([feature_map[-y, x] for x, y in pixels])

    return features

def set_obj_pos(sim, joint, pos=None, quat=None):
    pos = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(0.8, 1)]) if pos is None else pos
    quat = np.array([0, 0, 0, 0]) if quat is None else quat

    sim.data.set_joint_qpos(joint, np.concatenate([pos, quat]))


def to_pointcloud(sim, image, depth_map, camera):
    w, h = image.shape[1], image.shape[0]

    # pixel coordinates of which to sample world position
    # all_pixels = np.array([[x, y] for x in range(w) for y in range(h)
    # if image[y, x, 0] > 100/255 and image[y, x, 1] < 60/255 and image[y, x, 2] < 60/255])
    all_pixels = np.array([[x, y] for x in range(w) for y in range(h)])

    # TODO: filter out background


    # transformation matrix (pixel coord -> world coord) TODO: optimizable without inverse?
    world_to_pix = camera_utils.get_camera_transform_matrix(sim, camera, h, w)
    pix_to_world = np.linalg.inv(world_to_pix)

    points = pixel_to_world(all_pixels, depth_map, pix_to_world)
    rgb = pixel_to_feature(all_pixels, image)

    return points, rgb

def save_pointcloud(sim, image, depth_map, camera, file='pointcloud.npz'):
    points, rgb = to_pointcloud(sim, image, depth_map, camera)
    np.savez(file, points=points, rgb=rgb)


def random_action(env):
    return np.random.randn(env.robots[0].dof)



class UI:
    def __init__(self, window, camera_mover):
        self.window = window
        self.camera = camera_mover
        
        # mouse & keyboard states
        self.mouse_x = self.mouse_y = 0
        self.mouse_clicked = False
        self.key = None

        # mouse helpers
        self._mx, self._my = 0, 0
        self._mx_prev, self._my_prev = self._mx, self._my

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, lambda e, x, y, f, p: self.mouse_callback(e, x, y, f, p))
    
    def update(self):
        # update key state
        self.key = cv2.waitKey(10)

        # update mouse state
        if self.mouse_clicked:
            self._mx_prev, self._my_prev = self._mx, self._my
        self._mx, self._my = self.mouse_x, self.mouse_y


        if self.key == 27: #ESC key
            return False

        # move camera with WASD
        self.camera.move_camera((1,0,0), ((self.key == ord('d')) - (self.key == ord('a'))) * 0.05) # x axis (right-left)
        self.camera.move_camera((0,1,0), ((self.key == ord('2')) - (self.key == ord('z'))) * 0.05) # y axis (up-down)
        self.camera.move_camera((0,0,1), ((self.key == ord('s')) - (self.key == ord('w'))) * 0.05) # z axis (forward-backward)

        # rotate camera with mouse        
        if self.mouse_clicked:
            self.camera.rotate_camera(point=None, axis=(0, 1, 0), angle=(self._mx-self._mx_prev)/10)
            self.camera.rotate_camera(point=None, axis=(1, 0, 0), angle=(self._my-self._my_prev)/10)
        
        return True
    
    def is_pressed(self, key):
        if type(key) is str:
            return self.key == ord(key)
        else:
            return self.key == key

    def show(self, img):
        cv2.imshow(self.window, img)
    
    def close(self):
        cv2.destroyWindow(self.window)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_clicked = False
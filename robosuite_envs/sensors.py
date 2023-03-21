import numpy as np
from robosuite.utils.camera_utils import CameraMover
from abc import ABC, abstractmethod

class Sensor(ABC):
    '''
    Abstract base class for a sensor module
    It is essentially a layer between the environment and the encoder,
    converting the state into a observation that can be encoded later

    Inheriting classes must implement the following:
        - observe(state): given the ground truth state, returns the observation dict

    Inheriting classes may implement the following (optional):
        - reset(): called when the environment is reset
        - env_kwargs: kwargs when initializing robosuite env, e.g. camera settings
    '''
    requires_vision = False

    def __init__(self, cameras={}, camera_size=(128, 128)):
        '''
        cameras: dict of camera names and their desired poses, if any
        camera_size: (width, height) of the camera
        '''
        self.robo_env = None # set by the environment after initialization
        self.cameras = list(cameras.keys())
        self.poses = list(cameras.values())
        self.camera_size = camera_size

        self.movers = [None for c in self.cameras] # [CameraMover] in the same order as self.cameras
    
    @property
    def env_kwargs(self):
        if len(self.cameras) > 0:
            return { # kwargs for the robosuite env, e.g. camera settings
                'use_camera_obs': True,
                'camera_names': self.cameras,
                'camera_widths': self.camera_size[0],
                'camera_heights': self.camera_size[1],
            }
        else:
            return {'use_camera_obs': False}

    def create_movers(self):
        self.movers = [CameraMover(self.robo_env, camera=c) for c in self.cameras]
        for mover, pose in zip(self.movers, self.poses):
            if pose is not None:
                pos, quat = pose
                mover.set_camera_pose(np.array(pos), np.array(quat))
    
    def reset(self):
        self.create_movers()
        #TODO: rerender and return new observation
    
    @abstractmethod
    def observe(self, state):
        '''
        Returns a observation dict for the given ground truth state
        '''
        pass


class GroundTruthSensor(Sensor):
    requires_vision = False

    def observe(self, state):
        return state

import numpy as np
from random import uniform
from copy import deepcopy
import robosuite as suite
from robosuite.controllers import load_controller_config

from .base_env import RobosuiteGoalEnv, render_goal, render_goal, assert_correctness
from .encoders import PassthroughEncoder
from .sensors import PassthroughSensor
from .utils import apply_preset, set_obj_pos, set_robot_pose, disable_rendering

# misc settings
keep_cam_pose = False # reset camera position after each reset

# keyward arguments for Robosuite environment to be created
robo_kwargs = {}
# Configuration for each scene
cfg_scene = {}
# Configuration for each task
cfg_task = {}


########## Base Configs ###############
robo_kwargs['Base'] = {
    'has_renderer': False,
    'has_offscreen_renderer': True,
    'render_gpu_device_id': 0,
    'reward_shaping': False, # sparse reward
    'ignore_done': True, # unlimited horizon (use gym's TimeLimit wrapper instead)
}
cfg_scene['Base'] = {
    # 'camera_size': (128, 128), # width, height
    'camera_size': (256, 256), # width, height
    'sample_points': 2048, # points in the point cloud
    'sampler': 'FPS', # sampling method: 'FPS' or 'RS'
    'cameras': { # name: (position, quaternion)
        'frontview': ([ 1.5, 0, 1], [0.53, 0.53, 0.46, 0.46]), # front
    },
    'bbox': [[-0.8, 0.8], [-0.8, 0.8], [0.5, 2.0]], # (x_min, x_max), (y_min, y_max), (z_min, z_max)
}
cfg_scene['Base_full'] = {
    'camera_size': (256, 256), # width, height
    'sample_points': 2048, # points in the point cloud
    'sampler': 'FPS', # sampling method: 'FPS' or 'RS'
    'cameras': { # name: (position, quaternion)
        'frontview': ([ 1.5, 0, 1], [0.53, 0.53, 0.46, 0.46]), # front
        'agentview': ([-0.15, -1.2, 2.3], [0.3972332, 0, 0, 0.9177177]), # left
        'birdview': ([-0.15, 1.2, 2.3], [0, 0.3972332, 0.9177177, 0]), # right
    },
    'bbox': [[-0.8, 0.8], [-0.8, 0.8], [0.5, 2.0]], # (x_min, x_max), (y_min, y_max), (z_min, z_max)
}
# cfg_scene['Base'] = cfg_scene['Base_full']



########## Table Scene ##########
robo_kwargs['Table'] = robo_kwargs['Base'] | {
    'env_name': 'Lift', # name of the Robosuite environment
    'robots': 'Panda', 
    'controller_configs': load_controller_config(default_controller="OSC_POSITION"),
}
cfg_scene['Table'] = cfg_scene['Base_full'] | {
    'scene': 'Table', # name of this configuration, used to look up other configs of the same env

    # class segmentation, the index corresponds to the label value (integer encoding)
    'classes': ['env', 'cube', 'arm', 'base', 'gripper'], # cube only exists because it is index 1, but there is no cube in the scene
    'states': [None, None, None, None, 'robot0_eef_pos'],
    'state_dim': [0, 0, 0, 0, 3], # 0 will be ignored
    'class_latent_dim': [0, 0, 0, 0, 3], # 0 will be ignored
    'class_colors': [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0.5], [0, 0.4, 0], [0, 0, 1]],
    'class_distribution': [0.3, 0, 0.4, 0.05, 0.05], #TODO: from generate_pc
}


########## Cube Scene ##########
robo_kwargs['Cube'] = robo_kwargs['Table']
cfg_scene['Cube'] = cfg_scene['Base_full'] | {
    'scene': 'Cube', # name of this configuration, used to look up other configs of the same env

    # class segmentation, the index corresponds to the label value (integer encoding)
    'classes': ['env', 'cube', 'arm', 'base', 'gripper'],
    'states': [None, 'cube_pos', None, None, 'robot0_eef_pos'], # corresponding state name
    'state_dim': [0, 3, 0, 0, 3], # 0 will be ignored
    'class_latent_dim': [0, 3, 7, 0, 3], # 0 will be ignored
    'class_colors': [[0, 0, 0], [1, 0, 0], [0.8, 0.8, 0.8], [0, 1, 0], [0, 0, 1]],
    'class_distribution': [0.3, 0.01, 0.4, 0.05, 0.05], #TODO: from generate_pc
}


########## PegInHole Scene ##########
robo_kwargs['PegInHole'] = robo_kwargs['Base'] | {
    'env_name': 'TwoArmPegInHole', # name of the Robosuite environment
    'robots': ['Panda', 'Panda'], 
    # 'controller_configs': load_controller_config(default_controller="OSC_POSE"),
}
cfg_scene['PegInHole'] = cfg_scene['Base'] | {
    'scene': 'PegInHole', # name of this configuration, used to look up other configs of the same env
    'cameras': { # name: (position, quaternion)
        'agentview': None,
    },
    'bbox': [[-0.8, 0.8], [-0.8, 0.8], [0.5, 2.0]], # (x_min, x_max), (y_min, y_max), (z_min, z_max)

    # class segmentation, the index corresponds to the label value (integer encoding)
    # TODO
}



############# Define Tasks #############
class RoboReach(RobosuiteGoalEnv):
    task = 'Reach'
    scene = 'Table'

    # spaces
    proprio_keys = [] # purposefully empty
    obs_keys = ['robot0_eef_pos'] # eef position
    goal_keys = ['robot0_eef_pos'] # eef position

    def __init__(
        self,
        render_mode=None,
        sensor=PassthroughSensor,
        encoder=PassthroughEncoder,
        require_segmentation=False,
        **kwargs
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_scene[self.scene])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (512, 512) # width, height

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs[self.scene],
            sensor=sensor(env=self, require_segmentation=require_segmentation),
            encoder=encoder(self, self.obs_keys, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal,
            **kwargs
        )
        if keep_cam_pose:
            self.reset_camera_poses=False

    @staticmethod
    def set_initial_state(robo_env, get_state):
        robo_env.clear_objects('cube')
        robo_env.sim.forward()

    # define environment functions
    @assert_correctness
    def desired_goal_state(self, state, rerender=False):
        # desired_state = deepcopy(state) # easy but slow
        desired_state = state.copy() # shallow copy
        desired_state['robot0_eef_pos'] = np.zeros(3) # important! new array
        desired_state['robot0_eef_pos'][0] = np.random.uniform(-0.2, 0.2)
        desired_state['robot0_eef_pos'][1] = np.random.uniform(-0.2, 0.2)
        desired_state['robot0_eef_pos'][2] = np.random.uniform(0.85, 1.2)

        if rerender:
            if self.simulate_goal: # simulated goal
                desired_state, succ = self.simulate_eef_pos(desired_state['robot0_eef_pos'])
                if not succ:
                    print('Warning: failed to reach the desired robot pos for the goal state imagination')
            else: # visualized goal
                raise NotImplementedError

        return desired_state

    # def check_success(self, achieved, desired, info, force_gt=False):
    #     axis = 1 if achieved.ndim == 2 else None # batched version or not
    #     if not force_gt and self.encoder.latent_encoding:
    #         threshold = self.encoder.latent_threshold
    #         return (np.abs(achieved - desired) <= threshold).all(axis=axis)
    #     else:
    #         return np.linalg.norm(achieved - desired, axis=axis) < 0.05
        


########## Push ##########
class RoboPush(RobosuiteGoalEnv):
    task = 'Push'
    scene = 'Cube'

    # spaces
    proprio_keys = ['robot0_proprio-state'] # all proprioception
    obs_keys = ['cube_pos'] # observe the cube position
    goal_keys = ['cube_pos'] # we only care about cube position

    def __init__(
        self,
        render_mode=None,
        sensor=PassthroughSensor,
        encoder=PassthroughEncoder,
        require_segmentation=False,
        **kwargs
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_scene[self.scene])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (512, 512) # width, height

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs[self.scene],
            sensor=sensor(env=self, require_segmentation=require_segmentation),
            encoder=encoder(self, self.obs_keys, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal,
            simulate_goal=False, # robot pose is not relevant to goal state
            **kwargs
        )
        if keep_cam_pose:
            self.reset_camera_poses=False
 

    # define environment functions
    @assert_correctness
    def desired_goal_state(self, state, rerender=False):
        cube_pos = state['cube_pos'].copy()
        # pick random dist and direction to move the cube towards
        dist = np.random.uniform(0.13, 0.3) # move by at least 13cm so goal is not achieved by default
        dir = np.random.uniform(0, 2*np.pi)
        cube_pos[0] += dist * np.cos(dir)
        cube_pos[1] += dist * np.sin(dir)

        if rerender:
            if self.simulate_goal: # simulated goal
                raise NotImplementedError
            else: # visualized goal
                desired_state = self.render_state(lambda env: set_obj_pos(env.sim, joint='cube_joint0', pos=cube_pos))
        else:
            desired_state = state.copy()
            desired_state['cube_pos'] = cube_pos

        return desired_state

    # def check_success(self, achieved, desired, info, force_gt=False):
    #     axis = 1 if achieved.ndim == 2 else None # batched version or not
    #     if not force_gt and self.encoder.latent_encoding:
    #         threshold = self.encoder.latent_threshold
    #         return (np.abs(achieved - desired) <= threshold).all(axis=axis)
    #     else:
    #         return np.linalg.norm(achieved - desired, axis=axis) < 0.05
    
    def randomize(self):
        set_obj_pos(self.robo_env.sim, joint='cube_joint0', pos=np.array([uniform(-0.4, 0.4), uniform(-0.4, 0.4), uniform(0.8, 0.9)]))


    
########## Pick and Place ##########
class RoboPickAndPlace(RobosuiteGoalEnv):
    task = 'PickAndPlace'
    scene = 'Cube'

    # spaces
    proprio_keys = ['robot0_proprio-state'] # all proprioception
    obs_keys = ['cube_pos'] # observe the cube position
    goal_keys = ['cube_pos'] # we only care about cube position

    def __init__(
        self,
        render_mode=None,
        sensor=PassthroughSensor,
        encoder=PassthroughEncoder,
        require_segmentation=False,
        **kwargs
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_scene[self.scene])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (512, 512) # width, height

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs[self.scene],
            sensor=sensor(env=self, require_segmentation=require_segmentation),
            encoder=encoder(self, self.obs_keys, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal,
            **kwargs
        )
        if keep_cam_pose:
            self.reset_camera_poses=False
 

    # define environment functions
    @assert_correctness
    def desired_goal_state(self, state, rerender=False):
        # goal is the cube position
        cube_pos = state['cube_pos'].copy() # cube position
        # pick random dist and direction to move the cube towards
        dist = np.random.uniform(0.13, 0.2) # move by at least 13cm so goal is not achieved by default
        dir = np.random.uniform(0, 2*np.pi)
        cube_pos[0] += dist * np.cos(dir)
        cube_pos[1] += dist * np.sin(dir)
        if np.random.uniform() < 0.5: # cube in the air for 50% of the time
            cube_pos[2] += np.random.uniform(0.01, 0.2)

        if rerender:
            if self.simulate_goal: # simulated goal
                raise NotImplementedError
            else: # rendered goal
                # print('rerendering goal')
                desired_state = self.render_state(lambda env: set_obj_pos(env.sim, joint='cube_joint0', pos=cube_pos))
        else:
            desired_state = state.copy()
            desired_state['cube_pos'] = cube_pos

        return desired_state

    # def check_success(self, achieved, desired, info, force_gt=False):
    #     axis = 1 if achieved.ndim == 2 else None # batched version or not
    #     if not force_gt and self.encoder.latent_encoding:
    #         threshold = self.encoder.latent_threshold
    #         return (np.abs(achieved - desired) <= threshold).all(axis=axis)
    #     else:
    #         return np.linalg.norm(achieved - desired, axis=axis) < 0.05

    def randomize(self):
        set_obj_pos(self.robo_env.sim, joint='cube_joint0', pos=np.array([uniform(-0.4, 0.4), uniform(-0.4, 0.4), uniform(0.8, 1.3)]))




########## PegInHole ##########
class RoboPegInHole(RobosuiteGoalEnv):
    task = 'PegInHole'
    scene = 'PegInHole'

    # spaces
    # proprio_keys = ['robot0_proprio-state', 'robot1_proprio-state'] # all proprioception
    proprio_keys = [] # hard version, since peg and hole and basically eefs
    obs_keys = ['peg_to_hole', 'peg_quat', 'hole_pos', 'hole_quat'] # observe the pegs and holes
    goal_keys = ['peg_to_hole', 'hole_pos'] # we only care about cube position

    def __init__(
        self,
        render_mode=None,
        sensor=PassthroughSensor,
        encoder=PassthroughEncoder,
        require_segmentation=False,
        **kwargs
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_scene[self.scene])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (512, 512) # width, height

        def render_peg_hole(env, robo_obs):
            peg_pos = robo_obs['hole_pos'] - robo_obs['peg_to_hole']
            hole_pos = robo_obs['hole_pos']
            # print('peg_pos', peg_pos)
            return np.array([peg_pos, hole_pos]), np.array([[0, 1, 0], [1, 0, 0]])


        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs[self.scene],
            sensor=sensor(env=self, require_segmentation=require_segmentation),
            encoder=encoder(self, self.obs_keys, self.goal_keys),
            render_mode=render_mode,
            render_info=render_peg_hole,
            simulate_goal=False, # robot pose is not relevant to goal state
            **kwargs
        )
        if keep_cam_pose:
            self.reset_camera_poses=False
 

    # define environment functions
    @assert_correctness
    def desired_goal_state(self, state, rerender=False):
        return state

    # def check_success(self, achieved, desired, info, force_gt=False):
    #     axis = 1 if achieved.ndim == 2 else None # batched version or not
    #     if not force_gt and self.encoder.latent_encoding:
    #         threshold = self.encoder.latent_threshold
    #         return (np.abs(achieved - desired) <= threshold).all(axis=axis)
    #     else:
    #         return np.linalg.norm(achieved - desired, axis=axis) < 0.05



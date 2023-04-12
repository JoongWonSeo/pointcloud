import numpy as np
from random import uniform
import robosuite as suite
from robosuite.controllers import load_controller_config

from .base_env import RobosuiteGoalEnv
from .encoders import PassthroughEncoder
from .sensors import PassthroughSensor
from .utils import apply_preset, set_obj_pos, set_robot_pose, disable_rendering

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
    'camera_size': (128, 128), # width, height
    'sample_points': 2048, # points in the point cloud
}


########## Table Scene ##########
robo_kwargs['Table'] = robo_kwargs['Base'] | {
    'env_name': 'Lift', # name of the Robosuite environment
    'robots': 'Panda', 
    'controller_configs': load_controller_config(default_controller="OSC_POSITION"),
}
cfg_scene['Table'] = cfg_scene['Base'] | {
    'scene': 'Table', # name of this configuration, used to look up other configs of the same env
    'cameras': { # name: (position, quaternion)
        'frontview': ([-0.15, -1.2, 2.3], [0.3972332, 0, 0, 0.9177177]),
        'agentview': ([-0.15, 1.2, 2.3], [0, 0.3972332, 0.9177177, 0]),
        'birdview': ([1.5, 0, 2], [0.35629062, 0.35629062, 0.61078392, 0.61078392])
    },
    'bbox': [[-0.8, 0.8], [-0.8, 0.8], [0.5, 2.0]], # (x_min, x_max), (y_min, y_max), (z_min, z_max)

    # class segmentation, the index corresponds to the label value (integer encoding)
    # cube only exists because it is index 1, but there is no cube in the scene
    'classes': ['env', 'cube', 'arm', 'base', 'gripper'],
    'class_colors': [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1]],
    'class_distribution': [0.3, 0, 0.05, 0.05], #TODO: from generate_pc
    'class_gt_dim': [0, 0, 7+7, 0, 3], # 0 will be ignored
    'class_latent_dim': [0, 0, 7, 0, 3], # 0 will be ignored
}


########## Cube Scene ##########
# Configs for Envs based on the Robosuite Lift task
robo_kwargs['Cube'] = robo_kwargs['Table']
cfg_scene['Cube'] = cfg_scene['Table'] | {
    'scene': 'Cube', # name of this configuration, used to look up other configs of the same env

    # class segmentation, the index corresponds to the label value (integer encoding)
    'classes': ['env', 'cube', 'arm', 'base', 'gripper'],
    'class_colors': [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1]],
    'class_distribution': [0.3, 0.01, 0.4, 0.05, 0.05], #TODO: from generate_pc
    'class_gt_dim': [0, 3, 7+7, 0, 3], # 0 will be ignored
    'class_latent_dim': [0, 3, 7, 0, 3], # 0 will be ignored
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
        **kwargs
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_scene[self.scene])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (512, 512) # width, height

        # for visualization of the goal
        def render_goal(env, robo_obs):
            eef_goal = env.episode_goal_state['robot0_eef_pos']
            return np.array([eef_goal]), np.array([[0, 1, 0]])
            
        def render_pred(env, robo_obs):
            predicted_eef = env.encoding
            eef_goal = env.episode_goal_state['robot0_eef_pos']
            return np.array([predicted_eef, eef_goal]), np.array([[1, 0, 0], [0, 1, 0]])

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs[self.task],
            sensor=sensor(env=self),
            encoder=encoder(self, self.obs_keys, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal,
            **kwargs
        )

    @staticmethod
    def set_initial_state(robo_env, get_state):
        robo_env.clear_objects('cube')
        robo_env.sim.forward()

    # define environment functions
    def desired_goal_state(self, state, rerender=False):
        desired_state = state.copy() # shallow copy
        desired_state['robot0_eef_pos'][0] = np.random.uniform(-0.2, 0.2)
        desired_state['robot0_eef_pos'][1] = np.random.uniform(-0.2, 0.2)
        desired_state['robot0_eef_pos'][2] = np.random.uniform(0.85, 1.2)

        if rerender:
            if self.goal_env: # simulated goal
                desired_state, succ = self.simulate_eef_pos(desired_state['robot0_eef_pos'])
                if not succ:
                    print('Warning: failed to reach the desired robot pos for the goal state imagination')
            else: # visualized goal
                raise NotImplementedError

        return desired_state

    def check_success(self, achieved, desired, info):
        axis = 1 if achieved.ndim == 2 else None # batched version or not
        if self.encoder.latent_encoding:
            # threshold = np.array([0., 0., 0., 0., 0., 1.4301205, 1.6259564, 1.8243289, 0., 0., 0., 4.081347, 1.8291509, 0., 0.14140771, 0.], dtype=np.float32)
            threshold = np.array([0., 0., 0., 0., 0., 1.2245406, 1.588274, 1.9326254, 0., 0., 0., 3.5028644, 1.2349489, 0., 0.26512176, 0.], dtype=np.float32)
            #TODO: threshold should be encoder specific...
            # self.goal_encoder.goal_threshold * 0.05?

            return (np.abs(achieved - desired) <= threshold).all(axis=axis)
        else:
            return np.linalg.norm(achieved - desired, axis=axis) < 0.05
        


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
        **kwargs
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_scene[self.scene])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (512, 512) # width, height

        # for cube-only goal
        def render_goal_obs(env, robo_obs):
            encoded_cube = env.encoding
            cube_goal = env.episode_goal_encoding
            return np.array([encoded_cube, cube_goal]), np.array([[0, 1, 0], [1, 0, 0]])

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs[self.task],
            sensor=sensor(env=self),
            encoder=encoder(self, self.obs_keys, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal_obs,
            **kwargs
        )
 

    # define environment functions
    def desired_goal_state(self, state, rerender=False):
        cube_pos = state['cube_pos'].copy()
        # pick random dist and direction to move the cube towards
        dist = np.random.uniform(0.13, 0.2) # move by at least 13cm so goal is not achieved by default
        dir = np.random.uniform(0, 2*np.pi)
        cube_pos[0] += dist * np.cos(dir)
        cube_pos[1] += dist * np.sin(dir)

        if rerender:
            if self.goal_env: # simulated goal
                raise NotImplementedError
            else: # visualized goal
                print('rerendering goal')
                desired_state = self.render_state(lambda env: set_obj_pos(env.sim, joint='cube_joint0', pos=cube_pos))
        else:
            desired_state = state.copy()
            desired_state['cube_pos'] = cube_pos

        return desired_state

    def check_success(self, achieved, desired, info):
        axis = 1 if achieved.ndim == 2 else None # batched version or not
        return np.linalg.norm(achieved - desired, axis=axis) < 0.05
    
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
        **kwargs
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_scene[self.scene])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (512, 512) # width, height
        
        # for visualization of the goal
        def render_goal_obs(env, robo_obs):
            encoded_cube = env.encoding
            cube_goal = env.episode_goal_encoding
            return np.array([encoded_cube, cube_goal]), np.array([[0, 1, 0], [1, 0, 0]])

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs[self.task],
            sensor=sensor(env=self),
            encoder=encoder(self, self.obs_keys, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal_obs,
            **kwargs
        )
 

    # define environment functions
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
            if self.goal_env: # simulated goal
                raise NotImplementedError
            else: # rendered goal
                print('rerendering goal')
                desired_state = self.render_state(lambda env: set_obj_pos(env.sim, joint='cube_joint0', pos=cube_pos))
        else:
            desired_state = state.copy()
            desired_state['cube_pos'] = cube_pos

        return desired_state

    def check_success(self, achieved, desired, info):
        axis = 1 if achieved.ndim == 2 else None # batched version or not
        return np.linalg.norm(achieved - desired, axis=axis) < 0.05

    def randomize(self):
        set_obj_pos(self.robo_env.sim, joint='cube_joint0', pos=np.array([uniform(-0.4, 0.4), uniform(-0.4, 0.4), uniform(0.8, 1.3)]))



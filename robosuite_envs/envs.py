import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

from .base_env import RobosuiteGoalEnv
from .encoders import GroundTruthEncoder
from .sensors import GroundTruthSensor
from .utils import apply_preset

robo_kwargs = {} # keyward arguments for Robosuite environment to be created
cfg_vision = {} # RobosuiteGoalEnv attributes to be set for vision-based sensors


# Configs for Envs based on the Robosuite Lift task
robo_kwargs['Lift'] = {
    'env_name': "Lift", # try with other tasks like "Stack" and "Door"
    'robots': "Panda",  # try with other robots like "Sawyer" and "Jaco"
    'has_renderer': False,
    'has_offscreen_renderer': True,
    'render_gpu_device_id': 0,
    'controller_configs': load_controller_config(default_controller="OSC_POSITION"),
    'reward_shaping': False, # sparse reward
    'ignore_done': True, # unlimited horizon (use gym's TimeLimit wrapper instead)
}
cfg_vision['Lift'] = {
    'cameras': { # name: (position, quaternion)
        'frontview': ([0, -1.2, 1.8], [0.3972332, 0, 0, 0.9177177]),
        'agentview': ([0. , 1.2, 1.8], [0, 0.3972332, 0.9177177, 0]),
        'birdview': ([1.1, 0, 1.6], [0.35629062, 0.35629062, 0.61078392, 0.61078392])
    },
    'camera_size': (128, 128), # width, height
    'bbox': [[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]], # (x_min, x_max), (y_min, y_max), (z_min, z_max)
    'sample_points': 2048,
    'classes': [ # (name, RGB_for_visualization)
        ('env', [0, 0, 0]),
        ('cube', [1, 0, 0]),
        ('arm', [0.5, 0.5, 0.5]),
        ('base', [0, 1, 0]),
        ('gripper', [0, 0, 1]),
    ],
    'class_weights': [ # (name, weight) TODO: automatically compute weights
        ('env', 1.0),
        ('cube', 150.0),
        ('arm', 5.0),
        ('base', 10.0),
        ('gripper', 15.0),
    ]
}


class RobosuiteReach(RobosuiteGoalEnv):
    def __init__(
        self,
        render_mode=None,
        sensor=GroundTruthSensor,
        obs_encoder=GroundTruthEncoder,
        goal_encoder=GroundTruthEncoder
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_vision['Lift'])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (128, 128) # width, height

        # define proprioception, observation and goal keys
        self.proprio_keys = ['robot0_eef_pos'] # end-effector position
        self.obs_keys = [] # no observation
        self.goal_keys = ['cube_pos'] # goal is towards the cube

        # for visualization of the goal
        def render_goal(env, robo_obs):
            return np.array([env.episode_goal_encoding]), np.array([[0, 1, 0]])

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs['Lift'],
            proprio=self.proprio_keys,
            sensor=sensor(env=self),
            obs_encoder=obs_encoder(self, self.obs_keys),
            goal_encoder=goal_encoder(self, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal
        )

    # define environment feedback functions
    def achieved_goal(self, proprio, obs_encoding):
        return proprio # end-effector position
    
    def goal_state(self, state, rerender=False):
        desired_state = state.copy() # shallow copy
        desired_state['robot0_eef_pos'] = desired_state['cube_pos'] #  end-effector should be close to the cube

        if rerender:
            # create a dummy env, configure it to the desired state, and render it
            raise NotImplementedError('Rerendering is not implemented for this environment.')

        return desired_state

    def check_success(self, achieved, desired, info):
        # batched version
        if achieved.ndim == 2:
            return np.linalg.norm(achieved - desired, axis=1) < 0.05
        else: # single version
            return np.linalg.norm(achieved - desired) < 0.05
        




class RobosuiteLift(RobosuiteGoalEnv):
    def __init__(
        self,
        render_mode=None,
        sensor=GroundTruthSensor,
        obs_encoder=GroundTruthEncoder,
        goal_encoder=GroundTruthEncoder
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_vision['Lift'])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (128, 128) # width, height
        
        # define proprioception, observation and goal keys
        self.proprio_keys = ['robot0_eef_pos'] # end-effector position
        self.obs_keys = ['cube_pos'] # observe the cube position
        self.goal_keys = ['cube_pos'] # we only care about cube position

        # for visualization of the goal
        def render_goal(env, robo_obs):
            eef_goal = env.episode_goal_encoding[:3]
            cube_goal = env.episode_goal_encoding[3:]
            return np.array([eef_goal, cube_goal]), np.array([[1, 0, 0], [0, 1, 0]])
        
        def render_obs(env, robo_obs):
            # INEFFICIENT!
            encoded_cube = env.obs_encoder.encode(env.sensor.observe(robo_obs))
            return np.array([encoded_cube]), np.array([[0, 1, 0]])
        
        # for cube-only goal
        def render_goal_obs(env, robo_obs):
            encoded_cube = env.obs_encoder.encode(env.sensor.observe(robo_obs))
            cube_goal = env.episode_goal_encoding
            return np.array([encoded_cube, cube_goal]), np.array([[0, 1, 0], [1, 0, 0]])

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs['Lift'],
            proprio=self.proprio_keys,
            sensor=sensor(env=self),
            obs_encoder=obs_encoder(self, self.obs_keys),
            goal_encoder=goal_encoder(self, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal_obs
        )
 

    # define environment feedback functions
    def achieved_goal(self, proprio, obs_encoding):
        return obs_encoding # only cube position
    
    def goal_state(self, state, rerender=False):
        desired_state = state.copy() # shallow copy

        # goal is the cube position and end-effector position close to cube
        desired_state['cube_pos'] = state['cube_pos'].copy() # cube position
        desired_state['cube_pos'][2] += 0.2 # cube must be lifted up
        # desired_state['robot0_eef_pos'] = desired_state['cube_pos'].copy() # end-effector should be close to the cube (not really necessary)
        # desired_state['robot0_eef_pos'][2] += 0.05

        if rerender:
            raise NotImplementedError('Rerendering is not implemented for this environment.')
        
        return desired_state

    def check_success(self, achieved, desired, info):
        # TODO: experiment only check the cube position, ignore the end-effector position
        # also experiment with goal of moving the eef away from the cube

        # batched version
        if achieved.ndim == 2:
            return np.linalg.norm(achieved - desired, axis=1) < 0.1
        else: # single version
            return np.linalg.norm(achieved - desired) < 0.1

    


class RobosuitePickAndPlace(RobosuiteGoalEnv):
    def __init__(
        self,
        render_mode=None,
        sensor=GroundTruthSensor,
        obs_encoder=GroundTruthEncoder,
        goal_encoder=GroundTruthEncoder
        ):
        # configure cameras and their poses
        if sensor.requires_vision:
            apply_preset(self, cfg_vision['Lift'])
        else:
            # default camera with default pose
            self.cameras = {'frontview': None} if render_mode == 'human' else {}
            self.camera_size = (128, 128) # width, height
        
        # define proprioception, observation and goal keys
        self.proprio_keys = ['robot0_eef_pos'] # end-effector position
        self.obs_keys = ['cube_pos'] # observe the cube position
        self.goal_keys = ['cube_pos'] # we only care about cube position

        # for visualization of the goal
        def render_goal(env, robo_obs):
            eef_goal = env.episode_goal_encoding[:3]
            cube_goal = env.episode_goal_encoding[3:]
            return np.array([eef_goal, cube_goal]), np.array([[1, 0, 0], [0, 1, 0]])
        
        def render_obs(env, robo_obs):
            # INEFFICIENT!
            encoded_cube = env.obs_encoder.encode(env.sensor.observe(robo_obs))
            return np.array([encoded_cube]), np.array([[0, 1, 0]])
        
        # for cube-only goal
        def render_goal_obs(env, robo_obs):
            encoded_cube = env.obs_encoder.encode(env.sensor.observe(robo_obs))
            cube_goal = env.episode_goal_encoding
            return np.array([encoded_cube, cube_goal]), np.array([[0, 1, 0], [1, 0, 0]])

        # initialize RobosuiteGoalEnv
        super().__init__(
            robo_kwargs=robo_kwargs['Lift'],
            proprio=self.proprio_keys,
            sensor=sensor(env=self),
            obs_encoder=obs_encoder(self, self.obs_keys),
            goal_encoder=goal_encoder(self, self.goal_keys),
            render_mode=render_mode,
            render_info=render_goal_obs
        )
 

    # define environment feedback functions
    def achieved_goal(self, proprio, obs_encoding):
        return obs_encoding # only cube position
    
    def goal_state(self, state, rerender=False):
        desired_state = state.copy() # shallow copy

        # goal is the cube position and end-effector position close to cube
        desired_state['cube_pos'] = state['cube_pos'].copy() # cube position
         # cube must be moved
        desired_state['cube_pos'][0] += np.random.uniform(-0.2, 0.2)
        desired_state['cube_pos'][1] += np.random.uniform(-0.2, 0.2)
        if np.random.uniform() < 0.5: # cube in the air for 50% of the time
            desired_state['cube_pos'][2] += np.random.uniform(0.01, 0.2)

        if rerender:
            raise NotImplementedError('Rerendering is not implemented for this environment.')
        
        return desired_state

    def check_success(self, achieved, desired, info):
        # TODO: experiment only check the cube position, ignore the end-effector position
        # also experiment with goal of moving the eef away from the cube

        # batched version
        if achieved.ndim == 2:
            return np.linalg.norm(achieved - desired, axis=1) < 0.1
        else: # single version
            return np.linalg.norm(achieved - desired) < 0.1

    
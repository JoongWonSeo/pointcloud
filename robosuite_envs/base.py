# This wrapper class exposes a Gymnasium V26 conforming API for the Robosuite environment
# More specificially, it is to be used like a Gymnasium-Robotics GoalEnv (https://robotics.farama.org/content/multi-goal_api/)
# Meaning that the observation space is a dictionary with keys 'observation', 'desired_goal' and 'achieved_goal'

import numpy as np
import gymnasium as gym
from gymnasium_robotics.core import GoalEnv
from gymnasium.spaces import Box, Dict
import robosuite as suite
from robosuite.utils import camera_utils, transform_utils
from robosuite.utils.camera_utils import CameraMover
from .utils import UI, to_cv2_img, render

from abc import ABC, abstractmethod

# ObservationEncoder transforms the raw Robosuite observation into a single vector (i.e. image encoder or ground truth encoder)
class ObservationEncoder(ABC):
    '''
    Abstract class for robosuite observation encoders

    Inheriting classes must implement the following:
        - encode_state(self, observation): returns the selected proprioception and encoded state of the observation
        - encode_goal(self, observation): returns the encoded goal of the (initial) observation
        - get_space(self): observation space of the encoder (Gym Space)

    Inheriting classes may implement the following (optional):
        - env_kwargs: kwargs when initializing robosuite env, e.g. camera settings
        - reset(self, observation): called when the environment is reset
        - encode_proprioception(self, observation): returns the encoded proprioception of the observation
    '''

    def __init__(self, proprioception_keys, cameras, camera_size, robo_env=None):
        '''
        proprioception_keys: list of keys to select from the observation dict
        cameras: dict of camera name to their desired poses (position, rotation) if any
        camera_size: size of the camera image (width, height)
        robo_env: robosuite environment, can be overwritten by GoalEnvRobosuite in the constructor
        '''
        self.proprioception_keys = [proprioception_keys] if type(proprioception_keys) == str else list(proprioception_keys)
        self.robo_env = robo_env # this can be overwritten by GoalEnvRobosuite in the constructor
        self.cameras = cameras
        self.camera_size = camera_size
        self.camera_movers = None
    
    @property
    def env_kwargs(self):
        if len(self.cameras) > 0:
            return { # kwargs for the robosuite env, e.g. camera settings
                'use_camera_obs': True,
                'camera_names': list(self.cameras.keys()),
                'camera_widths': self.camera_size[0],
                'camera_heights': self.camera_size[1],
            }
        else:
            return {'use_camera_obs': False}
    
    def create_movers(self):
        self.camera_movers = [CameraMover(self.robo_env, camera=c) for c in self.cameras]
        for mover, c in zip(self.camera_movers, self.cameras):
            if self.cameras[c] is not None:
                pos, quat = self.cameras[c]
                mover.set_camera_pose(np.array(pos), np.array(quat))

    def reset(self, observation):
        self.create_movers()
        return self.encode(observation) #TODO: due to cameramovers, the actual is no longer same

    def encode(self, observation):
        return self.encode_proprioception(observation), self.encode_state(observation)

    def encode_proprioception(self, observation):
        obs_list = [observation[key] for key in self.proprioception_keys]
        if len(obs_list) > 0:
            return np.concatenate(obs_list, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)

    @abstractmethod
    def encode_state(self, observation):
        '''
        Returns the encoded state of the observation, excluding the proprioception
        '''
        pass

    @abstractmethod
    def encode_goal(self, observation):
        '''
        Returns the encoded goal of the (initial) observation
        '''
        pass

    @abstractmethod
    def get_space(self):
        '''
        Returns the observation space of the encoder
        '''
        pass


# GroundTruthEncoder returns the ground truth observation as a single vector
class GroundTruthEncoder(ObservationEncoder):
    def __init__(self, proprioception_keys, state_keys, goal_keys, robo_env=None):
        super().__init__(proprioception_keys, cameras={}, camera_size=(0, 0), robo_env=robo_env)
        self.state_keys = [state_keys] if type(state_keys) == str else list(state_keys)
        self.goal_keys = [goal_keys] if type(goal_keys) == str else list(goal_keys)
        self.all_keys = self.proprioception_keys + self.state_keys + self.goal_keys

    def encode_state(self, robo_obs):
        obs_list = [robo_obs[key] for key in self.state_keys]
        if len(obs_list) > 0:
            return np.concatenate(obs_list, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)
    
    def encode_goal(self, robo_obs):
        obs_list = [robo_obs[key] for key in self.goal_keys]
        if len(obs_list) > 0:
            return np.concatenate(obs_list, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)

    def get_space(self):
        # TODO: make into dict: https://robotics.farama.org/envs/fetch/slide/#observation-space
        o = self.robo_env.observation_spec()
        dim = sum([o[key].shape[0] for key in self.all_keys])
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(dim,))



# generic wrapper class around any robosuite environment
class RobosuiteGoalEnv(GoalEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, robo_env, encoder, achieved_goal, desired_goal, check_success, compute_reward=None, compute_truncated=None, compute_terminated=None, render_mode=None, render_info=None):
        '''
        robo_env: Robosuite environment
        achieved_goal: function that takes the *encoded* state+proprioception observations and returns the achieved goal
        desired_goal: function that takes the *initial Robosuite* observation and returns the desired goal
        check_success: function that takes (achieved_goal, desired_goal, info) and returns True if the task is completed
        compute_reward: function that takes (achieved_goal, desired_goal, info) and returns the reward
        compute_truncated: function that takes (achieved_goal, desired_goal, info) and returns True if the episode should be truncated
        compute_terminated: function that takes (achieved_goal, desired_goal, info) and returns True if the episode should be terminated
        encoder: ObservationEncoder that transforms the raw robosuite observation into a single vector
        render_mode: str for render mode such as 'human' or None
        '''
        ###################################################
        # internal variables, not part of the Gym Env API #
        ###################################################
        self.robo_env = robo_env
        self.check_success = check_success
        self.achieved_goal = lambda proprio, state: np.float32(achieved_goal(proprio, state))
        self.desired_goal = lambda robo_obs: np.float32(desired_goal(robo_obs))
        self.encoder = encoder
        if self.encoder.robo_env is None:
            self.encoder.robo_env = robo_env
        
        # information about the current episode
        self.is_episode_success = False
        self.episode_goal = None


        #######################
        # for Gym GoalEnv API #
        #######################
        # TODO: vectorize these functions for batched observations
        self.compute_reward = compute_reward or (lambda achieved_goal, desired_goal, info: self.check_success(achieved_goal, desired_goal, info) - 1)
        self.compute_truncated = compute_truncated or (lambda achieved_goal, desired_goal, info: self.robo_env.horizon == self.robo_env.timestep - 1)
        self.compute_terminated = compute_terminated or (lambda achieved_goal, desired_goal, info: False)

        ###################
        # for Gym Env API #
        ###################
        
        # setup attributes
        robo_obs = self.robo_env.observation_spec()
        proprio, state = self.encoder.encode(robo_obs)
        goal = self.achieved_goal(proprio, state) # both achieved and desired goal should be the same shape
        self.observation_space = Dict({
            'observation': self.encoder.get_space(),
            'achieved_goal': Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(goal.shape[0],)),
            'desired_goal': Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(goal.shape[0],)),
        })

        low, high = robo_env.action_spec
        self.action_space = Box(low=np.float32(low), high=np.float32(high))

        ###################
        # for rendering   #
        ###################
        self.encoder.create_movers()

        self.render_mode = render_mode
        self.render_info = render_info # function that returns points to render
        self.renderer = None
        self.request_truncate = False # from the UI


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.is_episode_success = False

        robo_obs = self.robo_env.reset()
        proprio, state = self.encoder.reset(robo_obs)
        obs = {
            'observation': np.concatenate((proprio, state)),
            'achieved_goal': self.achieved_goal(proprio, state),
            'desired_goal': self.desired_goal(robo_obs),
        }
        self.episode_goal = obs['desired_goal']
        info = {'is_success': self.is_episode_success}

        if self.render_mode == 'human':
            self.render_frame(robo_obs, info, reset=True)

        return obs, info
    

    def step(self, action):
        robo_obs, reward, done, info = self.robo_env.step(action)

        if self.episode_goal is None: # for if reset() is not called first
            self.episode_goal = self.desired_goal(robo_obs)
        
        proprio, state = self.encoder.encode(robo_obs)
        obs = {
            'observation': np.concatenate((proprio, state)),
            'achieved_goal': self.achieved_goal(proprio, state),
            'desired_goal': self.episode_goal,
        }
        if self.is_episode_success:
            info['is_success'] = True
        else:
            self.is_episode_success = self.check_success(obs['achieved_goal'], obs['desired_goal'], info)
            info['is_success'] = self.is_episode_success
        
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminated = self.compute_terminated(obs['achieved_goal'], obs['desired_goal'], info)
        truncated = done or self.compute_truncated(obs['achieved_goal'], obs['desired_goal'], info) or self.request_truncate

        if self.render_mode == 'human':
            self.render_frame(robo_obs, info)

        return obs, reward, terminated, truncated, info
    

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            

    def render_frame(self, robo_obs, info, reset=False):
        if self.render_mode is None:
            return

        if self.renderer is None: #init renderer
            self.renderer = UI('Robosuite', self.encoder)
        
        if reset: # dont render the first frame to avoid camera switch
            return
        
        # update UI
        if not self.renderer.update(): # if user closes the window
            quit() # force quit program
        self.request_truncate = self.renderer.is_pressed('r')

        # render
        cam = self.renderer.camera_name
        camera_image = robo_obs[cam + '_image'] / 255
        if self.render_info:
            camera_h, camera_w = camera_image.shape[:2]
            points, rgb = self.render_info(self, robo_obs)
            w2c = camera_utils.get_camera_transform_matrix(self.robo_env.sim, cam, camera_h, camera_w)
            render(points, rgb, camera_image, w2c, camera_h, camera_w)
        self.renderer.show(to_cv2_img(camera_image))
    
    




        



# class MultiGoalEnvironment(GymWrapper):
#     def __init__(self, env, check_success, achieved_goal, desired_goal, compute_reward=None, encoder=None, cameras=[], control_camera=None):
#         '''
#         env: the robosuite environment
#         compute_reward: the reward function to replace compute_reward(achieved_goal, desired_goal, info)
#         achieved_goal: the function that returns the achieved goal from current observation
#         desired_goal: the function that returns the desired goal from initial task state
#         encoder: PyTorch module the encodes the normalized 3D RGB point cloud to a fixed size vector
#         cameras: list of camera names to be used to generate the 3D RGB point cloud
#         control_camera: function that takes camera name and sets its pose
#         '''
#         self.check_success = check_success # function that returns True if the task is successful
#         self.achieved_goal = achieved_goal # function that returns the achieved goal from current observation
#         self.desired_goal = desired_goal # function that returns the desired goal from initial task state
#         if compute_reward is None:
#             self.compute_reward = lambda a, d, i: 0 if self.check_success(a, d, i) else -1 # reward function
#         self.encoder = encoder # encoder for the 3D RGB point cloud
#         self.cameras = cameras # list of camera names to be used to generate the 3D RGB point cloud
#         self.control_camera = control_camera # function that takes camera name and sets its pose

#         if encoder == None: # use ground-truth object states instead of encoder
#             # keys = ['object-state', 'robot0_proprio-state']
#             keys = ['cube_pos', 'robot0_eef_pos']
#         else: # use encoder to encode the 3D RGB point cloud
#             keys = [c+'_image' for c in cameras] + [c+'_depth' for c in cameras] + ['robot0_eef_pos']
#         super().__init__(env, keys=keys)

#         # task goal
#         self.current_goal = desired_goal(self._flatten_obs(env._get_observations()))

#         # Action limit for clamping: critically, assumes all dimensions share the same bound!
#         self.act_limit = self.action_space.high

#         # fix the observation dimension to include the goal
#         self.only_obs_dim = self.obs_dim # save the original observation dimension without the goal
#         self.only_goal_dim = self.current_goal.shape[0] # save the original goal dimension
#         if encoder == None:
#             self.obs_dim += self.only_goal_dim
#             high = np.inf * np.ones(self.obs_dim)
#             low = -high
#             self.observation_space = spaces.Box(low=low, high=high)
#         else:
#             # TODO
#             pass


#     # override the reset and step functions to include the goal
#     def reset(self):
#         obs = super().reset()
#         self.current_goal = self.desired_goal(obs)
#         return np.concatenate((obs, self.current_goal))

#     def step(self, action):
#         obs, reward, done, info = super().step(action)
#         reward = self.compute_reward(self.achieved_goal(obs), self.current_goal, info)
#         #done = True if reward >= 0 else False
#         #print('check_success() = ', self.env._check_success())
#         return np.concatenate((obs, self.current_goal)), reward, done, info

#     # good for HER: replace the desired goal with the achieved goal (virtual)
#     def replace_goal(self, obs, goal):
#         return np.concatenate((obs[:self.only_obs_dim], goal))

    
#     # get the 3D RGB point cloud of the current state, already merged, filtered and sampled to a fixed size
#     def get_pointcloud(self, cameras, filter_bbox, sample_size):
#         pass

#     # 
#     def get_camera_image(self, camera):
#         return self.env._get_observations()[camera + '_image'] / 255
    

# def make_multigoal_lift(horizon = 1000):
#     from robosuite.controllers import load_controller_config
#     # load default controller parameters for Operational Space Control (OSC)
#     controller_config = load_controller_config(default_controller="OSC_POSE")

#     env = suite.make(
#         env_name="Lift", # try with other tasks like "Stack" and "Door"
#         robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#         controller_configs=controller_config,
#         reward_shaping=False, # sparse reward
#         horizon=horizon,
#     )


#     # create a initial state to task goal mapper, specific to the task
#     def desired_goal(obs):
#         # goal is the cube position and end-effector position close to cube
#         cube_pos = obs[0:3]
#         # add random noise to the cube position
#         cube_pos[0] += np.random.uniform(-0.3, 0.3)
#         cube_pos[1] += np.random.uniform(-0.3, 0.3)
#         cube_pos[2] += 0.1
#         eef_pos = cube_pos # end-effector should be close to the cube
#         goal = eef_pos
#         return goal

#     # create a state to goal mapper for HER, such that the input state safisfies the returned goal
#     def achieved_goal(obs):
#         # real end-effector position
#         eef_pos = obs[3:6]
#         goal = eef_pos
#         return goal

#     # create a state-goal to reward function (sparse)
#     def check_success(achieved, desired, info):
#         return np.linalg.norm(achieved - desired) < 0.08

#     return MultiGoalEnvironment(
#         env,
#         check_success=check_success,
#         achieved_goal=achieved_goal,
#         desired_goal=desired_goal,
#     )
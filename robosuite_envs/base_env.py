# This wrapper class exposes a Gymnasium V26 conforming API for the Robosuite environment
# More specificially, it is to be used like a Gymnasium-Robotics GoalEnv (https://robotics.farama.org/content/multi-goal_api/)
# Meaning that the observation space is a dictionary with keys 'observation', 'desired_goal' and 'achieved_goal'

from abc import abstractmethod
import numpy as np
from gymnasium_robotics.core import GoalEnv
from gymnasium.spaces import Box, Dict
from robosuite.utils import camera_utils
from .encoders import GroundTruthEncoder, ObservationEncoder
from .utils import UI, to_cv2_img, render


# generic wrapper class around any robosuite environment
class RobosuiteGoalEnv(GoalEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, robo_env, proprio, sensor, obs_encoder, goal_encoder, render_mode=None, render_info=None):
        '''
        robo_env: Robosuite environment
        proprio: list of keys for the proprioception
        sensor: Sensor that transforms the ground truth into an observation (T -> O)
        obs_encoder: ObservationEncoder that transforms an observation into an encoding (O -> E)
        goal_encoder: ObservationEncoder that transforms an observation into a goal encoding (O -> G)
        render_mode: str for render mode such as 'human' or None
        render_info: a function that returns array of points and colors to render
        '''
        ###################################################
        # internal variables, not part of the Gym Env API #
        ###################################################
        self.robo_env = robo_env
        self.proprio = GroundTruthEncoder(proprio)
        self.sensor = sensor
        self.obs_encoder = obs_encoder
        self.goal_encoder = goal_encoder
        if self.sensor.robo_env is None:
            self.sensor.robo_env = self.robo_env
        
        # information about the current episode
        self.is_episode_success = False
        self.episode_goal = None

        #######################
        # for Gym GoalEnv API #
        #######################
        self.observation_space = Dict({
            'observation': ObservationEncoder.concat_spaces(self.robo_env, self.proprio, self.obs_encoder),
            'achieved_goal': self.goal_encoder.get_space(self.robo_env),
            'desired_goal': self.goal_encoder.get_space(self.robo_env),
        })
        self.action_space = Box(np.float32(robo_env.action_spec[0]), np.float32(robo_env.action_spec[1]), dtype=np.float32)

        ###################
        # for rendering   #
        ###################
        self.render_mode = render_mode
        self.render_info = render_info # function that returns points to render

        self.renderer = None
        self.request_truncate = False # from the UI
        self.sensor.create_movers()
    

    ###################################
    # defined by each individual task #
    ###################################
    @abstractmethod
    def achieved_goal(self, proprio, obs_encoding):
        '''
        function that takes the proprioception and observation encoding and returns the achieved goal in goal space
        E -> G
        '''
        pass

    @abstractmethod
    def goal_state(self, state, rerender=False):
        '''
        function that takes the *initial Robosuite* state and returns the desired goal state as a dict in state space
        T -> T
        '''
        pass

    @abstractmethod
    def check_success(self, achieved, desired, info) -> bool:
        '''
        function that takes (achieved_goal, desired_goal, info) and returns True if the task is completed
        '''
        pass


    #######################
    # for Gym GoalEnv API #
    #######################
    # TODO: vectorize these functions for batched observations
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.check_success(achieved_goal, desired_goal, info) - 1
    
    def compute_truncated(self, achieved_goal, desired_goal, info):
        return self.robo_env.horizon == self.robo_env.timestep - 1

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return False


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.is_episode_success = False

        # get the initial ground-truth state (S)
        state = self.robo_env.reset()
        self.sensor.reset()
        goal_state = self.goal_state(state, rerender=self.sensor.requires_vision)

        # convert the state into an observation (O)
        obs = self.sensor.observe(state)
        goal_obs = self.sensor.observe(goal_state)

        # proprioception does not need to be encoded
        proprio = self.proprio.encode(state)

        # encode the observation into the encoding space (E)
        obs_encoding = self.obs_encoder.encode(obs)
        goal_encoding = self.goal_encoder.encode(goal_obs)

        # create the observation dict for the agent
        peg = {
            'observation': np.concatenate((proprio, obs_encoding), dtype=np.float32),
            'achieved_goal': self.achieved_goal(proprio, obs_encoding),
            'desired_goal': goal_encoding,
        }

        # current episode information
        self.episode_goal = goal_encoding
        info = {'is_success': self.is_episode_success}

        if self.render_mode == 'human':
            self.render_frame(state, info, reset=True)

        return peg, info
    

    def step(self, action):
        # get the next ground-truth state (S)
        state, reward, done, info = self.robo_env.step(action)

        if self.episode_goal is None: # for if reset() is not called first
            goal_state = self.goal_state(state, rerender=self.sensor.requires_vision)
            goal_obs = self.sensor.observe(goal_state)
            self.episode_goal = self.goal_encoder.encode(goal_obs)
        
        # convert the state into an observation (O)
        obs = self.sensor.observe(state)

        # proprioception does not need to be encoded
        proprio = self.proprio.encode(state)

        # encode the observation into the encoding space (E)
        obs_encoding = self.obs_encoder.encode(obs)

        # create the observation dict for the agent (proprio, encoding, goal)
        peg = {
            'observation': np.concatenate((proprio, obs_encoding), dtype=np.float32),
            'achieved_goal': self.achieved_goal(proprio, obs_encoding),
            'desired_goal': self.episode_goal,
        }
        if self.is_episode_success:
            info['is_success'] = True
        else:
            self.is_episode_success = self.check_success(peg['achieved_goal'], peg['desired_goal'], info)
            info['is_success'] = self.is_episode_success
        
        reward = self.compute_reward(peg['achieved_goal'], peg['desired_goal'], info)
        terminated = self.compute_terminated(peg['achieved_goal'], peg['desired_goal'], info)
        truncated = done or self.request_truncate or self.compute_truncated(peg['achieved_goal'], peg['desired_goal'], info)

        if self.render_mode == 'human':
            self.render_frame(state, info)

        return peg, reward, terminated, truncated, info
    

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            

    def render_frame(self, robo_obs, info, reset=False):
        if self.render_mode is None:
            return

        if self.renderer is None: #init renderer
            self.renderer = UI('Robosuite', self.sensor)
        
        # if reset: # dont render the first frame to avoid camera switch
        #     return
        
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
    
    

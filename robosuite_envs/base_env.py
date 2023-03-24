# This wrapper class exposes a Gymnasium V26 conforming API for the Robosuite environment
# More specificially, it is to be used like a Gymnasium-Robotics GoalEnv (https://robotics.farama.org/content/multi-goal_api/)
# Meaning that the observation space is a dictionary with keys 'observation', 'desired_goal' and 'achieved_goal'

from abc import abstractmethod
import numpy as np
from gymnasium_robotics.core import GoalEnv
from gymnasium.spaces import Box, Dict
import robosuite as suite
from robosuite.utils.camera_utils import CameraMover, get_camera_transform_matrix
from .encoders import GroundTruthEncoder, ObservationEncoder
from .utils import UI, to_cv2_img, render


# generic wrapper class around any robosuite environment
class RobosuiteGoalEnv(GoalEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, robo_kwargs, proprio, sensor, obs_encoder, goal_encoder, render_mode=None, render_info=None):
        '''
        robo_kwargs: keyward arguments for Robosuite environment to be created
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
        if not hasattr(self, 'cameras'):
            self.cameras = {}
            self.camera_size = (0, 0)
        self.poses = list(self.cameras.values())
        self.cameras = list(self.cameras.keys())

        if len(self.cameras) > 0:
            robo_kwargs |= { # kwargs for the robosuite env, e.g. camera settings
                'use_camera_obs': True,
                'camera_names': self.cameras,
                'camera_widths': self.camera_size[0],
                'camera_heights': self.camera_size[1],
            }
        else:
            robo_kwargs |= {'use_camera_obs': False}

        self.robo_env = suite.make(hard_reset=False, **(robo_kwargs | sensor.env_kwargs))
        self.proprio = GroundTruthEncoder(self, proprio) # proprioception does not need to be encoded
        self.sensor = sensor
        self.obs_encoder = obs_encoder
        self.goal_encoder = goal_encoder
        
        # cached information about the current episode that is not returned by step()
        self.raw_state = None # raw state from the Robosuite environment
        self.observation = None # observation from the sensor
        self.proprioception = None # proprioception from the robot
        self.encoding = None # encoding from the observation encoder
        self.episode_goal_state = None # raw goal state from goal_state() function
        self.episode_goal_encoding = None # encoding from the goal encoder
        self.is_episode_success = False

        #######################
        # for Gym GoalEnv API #
        #######################
        self.observation_space = Dict({
            'observation': ObservationEncoder.concat_spaces(self.robo_env, self.proprio, self.obs_encoder),
            'achieved_goal': self.goal_encoder.get_space(self.robo_env),
            'desired_goal': self.goal_encoder.get_space(self.robo_env),
        })
        low, high = np.float32(self.robo_env.action_spec[0]), np.float32(self.robo_env.action_spec[1])
        self.action_space = Box(low, high, dtype=np.float32)

        ###################
        # for rendering   #
        ###################
        self.render_mode = render_mode
        self.render_info = render_info # function that returns points to render

        self.viewer = None
        self.request_truncate = False # from the UI

        # create CameraMovers and set their initial poses
        self.movers = [CameraMover(self.robo_env, camera=c) for c in self.cameras]
        self.set_camera_poses()
        
    

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
        backup = self.robo_env._get_observations
        self.robo_env._get_observations = lambda force_update: None # hack to prevent Robosuite from rendering
        self.robo_env.reset()
        self.set_camera_poses() # reset the camera poses
        self.robo_env._get_observations = backup
        state = self.robo_env._get_observations(force_update=True)

        self.sensor.reset()

        goal_state = self.goal_state(state, rerender=self.goal_encoder.requires_vision)

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

        # cache current episode information
        self.raw_state = state
        self.observation = obs
        self.proprioception = proprio
        self.encoding = obs_encoding
        self.episode_goal_state = goal_state
        self.episode_goal_encoding = goal_encoding
        info = {'is_success': self.is_episode_success}

        if self.render_mode == 'human':
            self.show_frame(state, info)

        return peg, info
    

    def step(self, action):
        # get the next ground-truth state (S)
        state, reward, done, info = self.robo_env.step(action)

        if self.episode_goal_encoding is None: # for if reset() is not called first
            goal_state = self.goal_state(state, rerender=self.goal_encoder.requires_vision)
            goal_obs = self.sensor.observe(goal_state)
            
            # cache current episode information
            self.episode_goal_state = goal_state
            self.episode_goal_encoding = self.goal_encoder.encode(goal_obs)
        
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
            'desired_goal': self.episode_goal_encoding,
        }
        if self.is_episode_success:
            info['is_success'] = True
        else:
            self.is_episode_success = self.check_success(peg['achieved_goal'], peg['desired_goal'], info)
            info['is_success'] = self.is_episode_success
        
        reward = self.compute_reward(peg['achieved_goal'], peg['desired_goal'], info)
        terminated = self.compute_terminated(peg['achieved_goal'], peg['desired_goal'], info)
        truncated = done or self.request_truncate or self.compute_truncated(peg['achieved_goal'], peg['desired_goal'], info)

        # cache current episode information
        self.raw_state = state
        self.observation = obs
        self.proprioception = proprio
        self.encoding = obs_encoding

        if self.render_mode == 'human':
            self.show_frame(state, info)

        return peg, reward, terminated, truncated, info

    
    def render(self):
        pass
        # if self.render_mode == 'human':
        #     self.show_frame(self.raw_state, {})
    

    def close(self):
        self.robo_env.close()
        if self.viewer is not None:
            self.viewer.close()


    #################
    # for rendering #
    #################
    def set_camera_poses(self):
        for mover, pose in zip(self.movers, self.poses):
            if pose is not None:
                pos, quat = pose
                mover.set_camera_pose(np.array(pos), np.array(quat))
                
    def render_state(self, state_setter):
        '''
        Render the given robosuite env state without affecting the actual state
        useful for rendering goal states or any 'imaginary' states
        '''
        backup = self.robo_env.sim.get_state()

        # set the state and render
        state_setter(self.robo_env)
        self.robo_env.sim.forward() # propagate the state
        state = self.robo_env._get_observations(force_update=True)

        # restore the original state
        self.robo_env.sim.set_state(backup)

        return state

    def show_frame(self, robo_obs, info):
        if self.render_mode is None:
            return

        if self.viewer is None: #init renderer
            self.viewer = UI('Robosuite', self)
        
        # update UI
        if not self.viewer.update(): # if user closes the window
            quit() # force quit program
        self.request_truncate = self.viewer.is_pressed('r')

        # render
        cam = self.cameras[self.viewer.camera_index]
        camera_image = robo_obs[cam + '_image'] / 255
        if self.render_info:
            camera_h, camera_w = camera_image.shape[:2]
            points, rgb = self.render_info(self, robo_obs)
            w2c = get_camera_transform_matrix(self.robo_env.sim, cam, camera_h, camera_w)
            render(points, rgb, camera_image, w2c, camera_h, camera_w)
        self.viewer.show(to_cv2_img(camera_image))
    
    

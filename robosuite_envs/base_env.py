# This wrapper class exposes a Gymnasium V26 conforming API for the Robosuite environment
# More specificially, it is to be used like a Gymnasium-Robotics GoalEnv (https://robotics.farama.org/content/multi-goal_api/)
# Meaning that the observation space is a dictionary with keys 'observation', 'desired_goal' and 'achieved_goal'

from abc import abstractmethod
from copy import deepcopy
from functools import reduce, wraps
import numpy as np
from numpy.testing import assert_equal
from gymnasium_robotics.core import GoalEnv
from gymnasium.spaces import Box, Dict
import robosuite as suite
from robosuite.utils.camera_utils import CameraMover, get_camera_transform_matrix
from robosuite.controllers import load_controller_config
from .encoders import PassthroughEncoder, ObservationEncoder, flatten_observations, flatten_robosuite_space
from .utils import UI, to_cv2_img, render, disable_rendering


# generic wrapper class around any robosuite environment
class RobosuiteGoalEnv(GoalEnv):
    metadata = {"render_modes": ["human"]}

    # should be set by the subclass
    task, scene = None, None
    proprio_keys, obs_keys, goal_keys = None, None, None

    def __init__(self, robo_kwargs, sensor, encoder, render_mode=None, render_info=None, **kwargs):
        '''
        robo_kwargs: keyward arguments for Robosuite environment to be created
        sensor: Sensor that transforms the ground truth into an observation (S -> O)
        obs_encoder: ObservationEncoder that turns an observation into an encoding and goal (O -> ExG)
        render_mode: str for render mode such as 'human' or None
        render_info: a function that returns array of points and colors to render

        Optional kwargs:
        visual_goal: manual override for whether goal is rendered or not
        simulate_goal: manual override for whether goal is simulated or not
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
        self.sensor = sensor
        self.encoder = encoder

        # only to check for actual success
        self.gt = PassthroughEncoder(env=self, obs_keys=self.encoder.obs_keys, goal_keys=self.encoder.goal_keys)
        
        # visual_goal might have already been overriden by the encoder
        if not hasattr(self, 'visual_goal'):
            self.visual_goal = kwargs.get('visual_goal', self.encoder.requires_vision)

        # cached information about the current episode that is not returned by step()
        self.raw_state = None # raw state from the Robosuite environment
        self.observation = None # observation from the sensor
        self.proprioception = None # proprioception from the robot
        self.encoding = None # encoding from the observation encoder
        self.goal_state = None # raw goal state from goal_state() function
        self.goal_obs = None # goal observation from the sensor
        self.goal_encoding = None # encoding from the goal encoder
        self.believe_success = False # whether the obs encoder believes it has succeeded
        self.actual_success = False # whether the task is actually successful
        self.is_episode_success = False # whether at least 1 step was successful


        #######################
        # for Gym GoalEnv API #
        #######################
        # note that this observation space refers to what the RL agent sees, which consists of the proprioception, observation and goal
        self.observation_space = Dict({
            'observation': ObservationEncoder.concat_spaces(
                flatten_robosuite_space(self.robo_env, self.proprio_keys),
                self.encoder.get_encoding_space(self.robo_env)
            ),
            'achieved_goal': self.encoder.get_goal_space(self.robo_env),
            'desired_goal': self.encoder.get_goal_space(self.robo_env),
        })
        low, high = np.float32(self.robo_env.action_spec[0]), np.float32(self.robo_env.action_spec[1])
        self.action_space = Box(low, high, dtype=np.float32)


        ###################
        # for rendering   #
        ###################
        self.render_mode = render_mode
        self.render_info = render_info # function that returns points to render
        self.overlay = None # a function that returns transparent overlay to render on top of the camera image

        self.viewer = None
        self.request_truncate = False # from the UI

        # create CameraMovers and set their initial poses
        self.movers = [CameraMover(self.robo_env, camera=c) for c in self.cameras]
        self.reset_camera_poses = self.sensor.requires_vision
        self.set_camera_poses()


        # dummy environment for goal imagination
        self.simulate_goal = kwargs.get('simulate_goal', self.visual_goal and self.encoder.global_encoding)
        if self.simulate_goal:
            abs_controller = load_controller_config(default_controller="OSC_POSITION")
            abs_controller['control_delta'] = False # desired eef position is absolute
            self.goal_env = suite.make(hard_reset=False, **(robo_kwargs | sensor.env_kwargs | {'controller_configs': abs_controller}))
            self.goal_cam_movers = [CameraMover(self.goal_env, camera=c) for c in self.cameras]
            print('Created a second env for goal state imagination')
        else:
            self.goal_env = None
            # self.goal_cam_movers = None
        

    ###################################
    # defined by each individual task #
    ###################################
    @abstractmethod
    def desired_goal_state(self, state, rerender=False):
        '''
        function that takes the *initial Robosuite* state and returns the desired goal state as a dict in state space
        S -> S
        '''
        pass

    @abstractmethod
    def check_success(self, achieved, desired, info, force_gt=False) -> bool:
        '''
        function that takes (achieved_goal, desired_goal, info) and returns True if the task is completed
        G x G -> {0, 1}
        '''
        axis = 1 if achieved.ndim == 2 else None # batched version or not
        if not force_gt and self.encoder.latent_encoding:
            threshold = self.encoder.latent_threshold
            return (np.abs(achieved - desired) <= threshold).all(axis=axis)
        else:
            return np.linalg.norm(achieved - desired, axis=axis) < 0.05

    @staticmethod
    def set_initial_state(robo_env, get_state):
        '''
        get_state: function that returns the current state of the environment (essentially self.robo_env._get_observations())

        Called after reset and before getting the first observation.
        You could use this function to set the initial state of the task.
        Or even simulate a few steps to imagine a goal state, and then reset the env.

        This is static so that it can also be applied to the goal_env.
        '''
        pass

    @abstractmethod
    def randomize(self):
        '''
        randomize the environment state, for training data generation
        '''
        pass


    #######################
    # for Gym GoalEnv API #
    #######################
    def compute_reward(self, achieved_goal, desired_goal, info):
        '''GxG -> {-1, 0}'''
        return self.check_success(achieved_goal, desired_goal, info) - 1
    
    def compute_truncated(self, achieved_goal, desired_goal, info):
        return self.robo_env.horizon == self.robo_env.timestep - 1 #TODO: we use gym wrapper instead

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return False # our tasks are continuous


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # remember the camera pose for convenience
        if not self.reset_camera_poses:
            self.poses = [deepcopy(mover.get_camera_pose()) for mover in self.movers]

        # get the initial ground-truth state (S)
        with disable_rendering(self.robo_env) as renderer:
            self.robo_env.reset()
            self.set_camera_poses() # reset the camera poses
            self.set_initial_state(self.robo_env, get_state=renderer) # set the initial state of the robo env
            state = renderer(force_update=True)

        self.sensor.reset()

        goal_state = self.desired_goal_state(state, rerender=self.visual_goal)

        # convert the state into an observation (O)
        obs = self.sensor.observe(state)
        goal_obs = self.sensor.observe(goal_state)

        # encode the observation into the encoding space (E)
        proprio = flatten_observations(state, self.proprio_keys)
        obs_encoding, achieved_goal = self.encoder(obs)
        goal_encoding = self.encoder.encode_goal(goal_obs)

        # create the observation dict for the agent
        peg = {
            'observation': np.concatenate((proprio, obs_encoding), dtype=np.float32),
            'achieved_goal': achieved_goal,
            'desired_goal': goal_encoding,
        }

        # cache current episode information
        self.raw_state = state
        self.observation = obs
        self.proprioception = proprio
        self.encoding = obs_encoding
        self.goal_state = goal_state
        self.goal_obs = goal_obs
        self.goal_encoding = goal_encoding
        self.believe_success = self.check_success(achieved_goal, goal_encoding, None)
        self.actual_success = self.check_success(self.gt.encode_goal(state), self.gt.encode_goal(goal_state), None, force_gt=True)
        self.is_episode_success = self.believe_success
        info = {'is_success': self.is_episode_success}

        if self.render_mode == 'human':
            self.show_frame(state, info)

        return peg, info
    

    def step(self, action):
        # get the next ground-truth state (S)
        state, reward, done, info = self.robo_env.step(action)

        if self.goal_encoding is None: # for if reset() is not called first
            goal_state = self.desired_goal_state(state, rerender=self.visual_goal)
            goal_obs = self.sensor.observe(goal_state)
            
            # cache current episode information
            self.goal_state = goal_state
            self.goal_obs = goal_obs
            self.goal_encoding = self.encoder.encode_goal(goal_obs)
        
        # convert the state into an observation (O)
        obs = self.sensor.observe(state)

        # encode the observation into the encoding space (E)
        proprio = flatten_observations(state, self.proprio_keys)
        obs_encoding, achieved_goal = self.encoder(obs)

        # create the observation dict for the agent (proprio, encoding, goal)
        peg = {
            'observation': np.concatenate((proprio, obs_encoding), dtype=np.float32),
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal_encoding,
        }
        
        self.believe_success = self.check_success(achieved_goal, self.goal_encoding, None)
        self.actual_success = self.check_success(self.gt.encode_goal(state), self.gt.encode_goal(self.goal_state), None, force_gt=True)

        if self.is_episode_success:
            info['is_success'] = True
        else:
            self.is_episode_success = self.believe_success
            info['is_success'] = self.is_episode_success
        
        reward = self.compute_reward(achieved_goal, self.goal_encoding, info)
        terminated = self.compute_terminated(achieved_goal, self.goal_encoding, info)
        truncated = done or self.request_truncate or self.compute_truncated(achieved_goal, self.goal_encoding, info)

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
        if self.goal_env is not None:
            self.goal_env.close()
    
    def __del__(self):
        self.close()


    #################
    # for rendering #
    #################
    def set_camera_poses(self, movers=None, poses=None):
        movers = movers or self.movers
        poses = poses or self.poses

        for mover, pose in zip(movers, poses):
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

            # show success info
            # left half of the bottom row is actual success
            mid = camera_w // 2
            camera_image[0:2, :mid, :] = [0, 1, 0] if self.actual_success else [1, 0, 0]
            # right half of the bottom row is believe success
            camera_image[0:2, mid:, :] = [0, 1, 0] if self.believe_success else [1, 0, 0]

        if self.overlay:
            camera_image += self.overlay(camera_h, camera_w)

        self.viewer.show(to_cv2_img(camera_image))
    

    def simulate_eef_pos(self, target, state_setter=None, tolerance=0.01, max_steps=50, eef_key='robot0_eef_pos'):
        if self.simulate_goal:
            success = False
            with disable_rendering(self.goal_env) as renderer:
                self.goal_env.reset()
                self.set_camera_poses(movers=self.goal_cam_movers)
                self.set_initial_state(self.goal_env, renderer)

            # try to move to the target
            action = np.zeros_like(self.goal_env.action_spec[0])
            action[0:3] = target
            for i in range(max_steps):
                state, reward, done, info = self.goal_env.step(action)
                if np.linalg.norm(state[eef_key] - target) < tolerance:
                    # print('early break at', i)
                    # print(state[eef_key], target)
                    success = True
                    break

            # if state_setter is provided, set the state to the given state
            if state_setter:
                state_setter(self.goal_env)
                self.goal_env.sim.forward()
            
            # return the state
            state = self.goal_env._get_observations(force_update=True)
            return state, success
        else:
            raise Exception('goal simulation is disabled')
            

################# Utils #################

# render infos
def render_goal(env, robo_obs):
    p, c = [], []
    
    # encoder produces GT prediction
    if env.encoder.requires_vision and not env.encoder.latent_encoding:
        # prediction
        p.append(env.encoding)
        c.append([1, 0, 0])
        
        # goal prediction
        p.append(env.goal_encoding)
        c.append([0, 0.7, 0])

    # goal
    p.append(env.goal_state[env.goal_keys[0]])
    c.append([0, 1, 0])

    return np.array(p), np.array(c)

# safety check
def assert_correctness(func):
    # disable correctness check for performance
    # return func

    if func.__name__ == 'desired_goal_state':
        # ensure initial state is not changed
        @wraps(func)
        def wrapper(*args, **kwargs):
            self, state = args[0], args[1]
            backup = deepcopy(state) # backup the initial state

            result = func(*args, **kwargs)
            assert_equal(state, backup)
            # print('Correctness check passed for', func.__name__)
            
            return result
        return wrapper            

    else:
        print('Warning: no correctness check for', func.__name__, 'implemented, skipping...')
        return func

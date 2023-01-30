# This wrapper class exposes a Gymnasium V26 conforming API for the Robosuite environment
# More specificially, it is to be used like a Gymnasium-Robotics GoalEnv (https://robotics.farama.org/content/multi-goal_api/)
# Meaning that the observation space is a dictionary with keys 'observation', 'desired_goal' and 'achieved_goal'

import numpy as np
import gymnasium as gym
from gymnasium_robotics.core import GoalEnv
from gymnasium.spaces import Box, Dict
import robosuite as suite
from robosuite.utils import camera_utils, transform_utils
from utils import UI

from abc import ABC, abstractmethod

# ObservationEncoder transforms the raw Robosuite observation into a single vector (i.e. image encoder or ground truth encoder)
class ObservationEncoder(ABC):
    @abstractmethod
    def encode(self, observation):
        pass

    @abstractmethod
    def get_space(self):
        pass


# GroundTruthEncoder returns the ground truth observation as a single vector
class GroundTruthEncoder(ObservationEncoder):
    def __init__(self, state_keys, proprioception_keys, robo_env = None):
        self.state_keys = [state_keys] if type(state_keys) == str else list(state_keys)
        self.proprioception_keys = [proprioception_keys] if type(proprioception_keys) == str else list(proprioception_keys)
        self.all_keys = self.state_keys + self.proprioception_keys
        self.robo_env = robo_env # this can be overwritten by GoalEnvRobosuite in the constructor

    def encode(self, obs):
        return self.encode_state(obs), self.encode_proprioception(obs)
    
    def encode_state(self, obs):
        obs_list = [obs[key] for key in self.state_keys]
        if len(obs_list) > 0:
            return np.concatenate(obs_list, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)

    def encode_proprioception(self, obs):
        obs_list = [obs[key] for key in self.proprioception_keys]
        if len(obs_list) > 0:
            return np.concatenate(obs_list, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)

    def get_space(self):
        o = self.robo_env.observation_spec()
        dim = sum([o[key].shape[0] for key in self.all_keys])
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(dim,))



# generic wrapper class around any robosuite environment
class RobosuiteGoalEnv(GoalEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, robo_env, achieved_goal, desired_goal, check_success, compute_reward=None, compute_truncated=None, compute_terminated=None, encoder=GroundTruthEncoder('object-state', 'robot0_proprio-state'), render_mode=None):
        '''
        robo_env: Robosuite environment
        achieved_goal: function that takes the *encoded* state+proprioception observations and returns the achieved goal
        desired_goal: function that takes the *raw Robosuite* observation and returns the desired goal
        check_success: function that takes (achieved_goal, desired_goal, info) and returns True if the task is completed
        compute_reward: function that takes (achieved_goal, desired_goal, info) and returns the reward
        compute_truncated: function that takes (achieved_goal, desired_goal, info) and returns True if the episode should be truncated
        compute_terminated: function that takes (achieved_goal, desired_goal, info) and returns True if the episode should be terminated
        encoder: ObservationEncoder that transforms the raw robosuite observation into a single vector
        render_mode: str for render mode such as 'human' or 'rgb_array'
        '''
        
        # internal variables, not part of the Gym Env API
        self._robo_env = robo_env
        self._check_success = check_success
        self._achieved_goal = lambda state, proprio: np.float32(achieved_goal(state, proprio))
        self._desired_goal = lambda robo_obs: np.float32(desired_goal(robo_obs))
        self._encoder = encoder
        if self._encoder.robo_env is None:
            self._encoder.robo_env = robo_env
        self._is_episode_success = False


        # for Gym GoalEnv API
        # TODO: vectorize these functions for batched observations
        self.compute_reward = compute_reward or (lambda achieved_goal, desired_goal, info: self._check_success(achieved_goal, desired_goal, info) - 1)
        self.compute_truncated = compute_truncated or (lambda achieved_goal, desired_goal, info: self._robo_env.horizon == self._robo_env.timestep - 1)
        self.compute_terminated = compute_terminated or (lambda achieved_goal, desired_goal, info: False)


        # for Gym Env API
        
        # setup attributes
        robo_obs = self._robo_env.observation_spec()
        state, proprio = self._encoder.encode(robo_obs)
        goal = self._achieved_goal(state, proprio) # both achieved and desired goal should be the same shape
        self.observation_space = Dict({
            'observation': self._encoder.get_space(),
            'achieved_goal': Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(goal.shape[0],)),
            'desired_goal': Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(goal.shape[0],)),
        })

        low, high = robo_env.action_spec
        self.action_space = Box(low=np.float32(low), high=np.float32(high))

        self.render_mode = render_mode
        self._renderer = None
        self._request_truncate = False # from the UI


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._is_episode_success = False

        robo_obs = self._robo_env.reset()
        state, proprio = self._encoder.encode(robo_obs)
        obs = {
            'observation': np.concatenate((state, proprio)),
            'achieved_goal': self._achieved_goal(state, proprio),
            'desired_goal': self._desired_goal(robo_obs),
        }
        info = {'is_success': self._is_episode_success}

        if self.render_mode == 'human':
            self._render_frame(robo_obs, info, reset=True)

        return obs, info
    

    def step(self, action):
        robo_obs, reward, done, info = self._robo_env.step(action)
        
        state, proprio = self._encoder.encode(robo_obs)
        obs = {
            'observation': np.concatenate((state, proprio)),
            'achieved_goal': self._achieved_goal(state, proprio),
            'desired_goal': self._desired_goal(robo_obs),
        }
        if self._is_episode_success:
            info['is_success'] = True
        else:
            self._is_episode_success = self._check_success(obs['achieved_goal'], obs['desired_goal'], info)
            info['is_success'] = self._is_episode_success
        
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminated = self.compute_terminated(obs['achieved_goal'], obs['desired_goal'], info)
        truncated = done or self.compute_truncated(obs['achieved_goal'], obs['desired_goal'], info) or self._request_truncate

        if self.render_mode == 'human':
            self._render_frame(robo_obs, info)

        return obs, reward, terminated, truncated, info
    

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            

    def _render_frame(self, robo_obs, info, reset=False):
        if self.render_mode is None:
            return
        
        if reset or self._renderer is None: # create camera mover
            if reset: # default camera pose
                pos, quat = [-0.2, -1.2, 1.8], transform_utils.axisangle2quat([0.817, 0, 0])
            else: #remember the camera pose
                pos, quat = self._camera.get_camera_pose()
            self._camera = camera_utils.CameraMover(self._robo_env, camera='agentview')
            self._camera.set_camera_pose(pos, quat)
            if self._renderer is not None:
                self._renderer.camera = self._camera

        if self._renderer is None: #init renderer
            self._renderer = UI('Robosuite', self._camera)
        
        if reset: # dont render the first frame to avoid camera switch
            return
        
        # update UI
        if not self._renderer.update(): # if user closes the window
            quit() # force quit program
        self._request_truncate = self._renderer.is_pressed('r')

        # render
        camera_image = robo_obs['agentview_image'] / 255
        # convert to CV2 format (flip along y axis and from RGB to BGR)
        camera_image = np.flip(camera_image, axis=0)
        camera_image = camera_image[:, :, [2, 1, 0]]

        self._renderer.show(camera_image)
    




        



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
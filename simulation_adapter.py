import numpy as np;
from gym import spaces


# This provides a layer of abstraction between the simulation API and the RL agent/training
# The API is meant to be somewhat gym-like

from robosuite.wrappers.gym_wrapper import GymWrapper

class MultiGoalEnvironment(GymWrapper):
    def __init__(self, env, reward, achieved_goal, desired_goal, encoder=None, cameras=[], control_camera=None):
        '''
        env: the robosuite environment
        reward: the reward function similar to compute_reward(achieved_goal, desired_goal, info)
        achieved_goal: the function that returns the achieved goal from current observation
        desired_goal: the function that returns the desired goal from initial task state
        encoder: PyTorch module the encodes the normalized 3D RGB point cloud to a fixed size vector
        cameras: list of camera names to be used to generate the 3D RGB point cloud
        control_camera: function that takes camera name and sets its pose
        '''
        self.reward = reward # reward function
        self.achieved_goal = achieved_goal # function that returns the achieved goal from current observation
        self.desired_goal = desired_goal # function that returns the desired goal from initial task state
        self.encoder = encoder # encoder for the 3D RGB point cloud
        self.cameras = cameras # list of camera names to be used to generate the 3D RGB point cloud
        self.control_camera = control_camera # function that takes camera name and sets its pose

        if encoder == None: # use ground-truth object states instead of encoder
            keys = ['object-state', 'robot0_proprio-state']
        else: # use encoder to encode the 3D RGB point cloud
            keys = [c+'_image' for c in cameras] + [c+'_depth' for c in cameras] + ['robot0_proprio-state']
        super().__init__(env, keys=keys)


        # task goal
        self.current_goal = desired_goal(self._flatten_obs(env._get_observations()))

        # fix the observation dimension to include the goal
        if encoder == None:
            self.obs_dim += self.current_goal.shape[0]
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            # TODO
            pass


    
    def reset(self):
        obs = super().reset()
        self.current_goal = self.desired_goal(obs)
        return np.concatenate((obs, self.current_goal))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = self.reward(self.achieved_goal(obs), self.current_goal, info)
        return np.concatenate((obs, self.current_goal)), reward, done, info

    
    # get the 3D RGB point cloud of the current state, already merged, filtered and sampled to a fixed size
    def get_pointcloud(self, cameras, filter_bbox, sample_size):
        pass

    # 
    def get_camera_image(self, camera):
        return self.env._get_observations()[camera + '_image'] / 255
    
    
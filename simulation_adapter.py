import numpy as np;
from gym import spaces
import robosuite as suite



# This provides a layer of abstraction between the simulation API and the RL agent/training
# The API is meant to be somewhat gym-like

from robosuite.wrappers.gym_wrapper import GymWrapper

class MultiGoalEnvironment(GymWrapper):
    def __init__(self, env, compute_reward, achieved_goal, desired_goal, encoder=None, cameras=[], control_camera=None):
        '''
        env: the robosuite environment
        compute_reward: the reward function to replace compute_reward(achieved_goal, desired_goal, info)
        achieved_goal: the function that returns the achieved goal from current observation
        desired_goal: the function that returns the desired goal from initial task state
        encoder: PyTorch module the encodes the normalized 3D RGB point cloud to a fixed size vector
        cameras: list of camera names to be used to generate the 3D RGB point cloud
        control_camera: function that takes camera name and sets its pose
        '''
        self.compute_reward = compute_reward # reward function
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

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.action_space.high

        # fix the observation dimension to include the goal
        self.only_obs_dim = self.obs_dim # save the original observation dimension without the goal
        self.only_goal_dim = self.current_goal.shape[0] # save the original goal dimension
        if encoder == None:
            self.obs_dim += self.only_goal_dim
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            # TODO
            pass


    # override the reset and step functions to include the goal
    def reset(self):
        obs = super().reset()
        self.current_goal = self.desired_goal(obs)
        return np.concatenate((obs, self.current_goal))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = self.compute_reward(self.achieved_goal(obs), self.current_goal, info)
        return np.concatenate((obs, self.current_goal)), reward, done, info

    # good for HER: replace the desired goal with the achieved goal (virtual)
    def replace_goal(self, obs, goal):
        return np.concatenate((obs[:self.only_obs_dim], goal))

    
    # get the 3D RGB point cloud of the current state, already merged, filtered and sampled to a fixed size
    def get_pointcloud(self, cameras, filter_bbox, sample_size):
        pass

    # 
    def get_camera_image(self, camera):
        return self.env._get_observations()[camera + '_image'] / 255
    

def make_multigoal_lift(horizon = 100):
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=False, # sparse reward
        horizon=horizon,
    )


    # create a initial state to task goal mapper, specific to the task
    def desired_goal(obs):
        goal = obs[:3]
        goal[2] += 0.05 # lift the object by 5cm (in the lift task, it's defined as 4cm above table height)
        return goal

    # create a state to goal mapper for HER, such that the input state safisfies the returned goal
    def achieved_goal(obs):
        return obs[:3] # object position

    # create a state-goal to reward function (sparse)
    def compute_reward(achieved, desired, info):
        # penalize for not reaching goal
        return 0 if np.linalg.norm(achieved - desired) < 0.04 else -1

    return MultiGoalEnvironment(
        env,
        compute_reward=compute_reward,
        achieved_goal=achieved_goal,
        desired_goal=desired_goal,
    )
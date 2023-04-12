from abc import ABC, abstractmethod

import numpy as np
from gymnasium.spaces import Box

# utils
def flatten_observations(obs, keys, dtype=np.float32):
    # gather and flatten the observation
    obs_list = [np.array(obs[key]).reshape((-1,)) for key in keys]
    return np.concatenate(obs_list, dtype=dtype) if len(obs_list) > 0 else np.array([], dtype=dtype)

def flatten_robosuite_space(robo_env, keys, low=-np.inf, high=np.inf, dtype=np.float32):
    o = robo_env.observation_spec()
    dim = sum([o[key].shape[0] if isinstance(o[key], np.ndarray) else 1 for key in keys])
    return Box(low=dtype(low), high=dtype(high), shape=(dim,))


# ObservationEncoder transforms the raw Robosuite observation into a single vector (i.e. image encoder or ground truth encoder)
class ObservationEncoder(ABC):
    '''
    Abstract class for observation encoders
    It essentially converts the sensor observation into an encoding (O -> E),
    which is then used by the agent to produce an action to perform the task.

    Inheriting classes must implement the following:
        - encode(self, observation): returns the encoded state of the given observation
        - get_space(self): observation space of the encoder (Gym Space)
    '''
    requires_vision = False # whether the encoder requires vision (rendering) or not
    latent_encoding = False # whether the encoder produces a latent encoding or the ground truth state space (either passthrough or predicted)
    dtype = np.float32

    def __init__(self, env, obs_keys, goal_keys):
        '''
        obs_keys: list of keys to select from the observation dict
        '''
        self.env = env
        self.obs_keys = [obs_keys] if type(obs_keys) == str else list(obs_keys)
        self.goal_keys = [goal_keys] if type(goal_keys) == str else list(goal_keys)

    @abstractmethod
    def encode_observation(self, observation):
        '''
        Returns the encoded state of the observation, excluding the proprioception
        '''
        pass
    
    @abstractmethod
    def encode_goal(self, observation):
        '''
        Returns the encoded state of the observation, excluding the proprioception
        '''
        pass

    @abstractmethod
    def get_encoding_space(self, robo_env):
        '''
        Returns the observation encoding space of the encoder
        '''
        pass

    @abstractmethod
    def get_goal_space(self, robo_env):
        '''
        Returns the goal encoding space of the encoder
        '''
        pass

    def __call__(self, observation):
        '''
        Returns the observation encoding and achieved goal encoding
        '''
        return self.encode_observation(observation), self.encode_goal(observation)

    @staticmethod
    def concat_spaces(*spaces):
        '''
        Concatenates the observation spaces of the given encoders
        '''
        lows = np.concatenate([s.low for s in spaces], axis=0)
        highs = np.concatenate([s.high for s in spaces], axis=0)
        return Box(lows, highs)


# PassthroughEncoder returns the ground truth observation as a single vector
class PassthroughEncoder(ObservationEncoder):
    requires_vision = False
    latent_encoding = False

    def encode_observation(self, obs):
        return flatten_observations(obs, self.obs_keys, self.dtype)
    
    def encode_goal(self, obs):
        return flatten_observations(obs, self.goal_keys, self.dtype)

    def get_encoding_space(self, robo_env):
        return flatten_robosuite_space(robo_env, self.obs_keys, dtype=self.dtype)
    
    def get_goal_space(self, robo_env):
        return flatten_robosuite_space(robo_env, self.goal_keys, dtype=self.dtype)
    


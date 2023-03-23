import numpy as np
from gymnasium.spaces import Box
from abc import ABC, abstractmethod

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

    def __init__(self, env, obs_keys):
        '''
        obs_keys: list of keys to select from the observation dict
        '''
        self.env = env
        self.obs_keys = [obs_keys] if type(obs_keys) == str else list(obs_keys)

    @abstractmethod
    def encode(self, observation):
        '''
        Returns the encoded state of the observation, excluding the proprioception
        '''
        pass

    @abstractmethod
    def get_space(self):
        '''
        Returns the observation space of the encoder
        '''
        pass
    
    @staticmethod
    def concat_spaces(robo_env, *encoders):
        '''
        Concatenates the observation spaces of the given encoders
        '''
        spaces = [e.get_space(robo_env) for e in encoders]
        lows = np.concatenate([s.low for s in spaces], axis=0)
        highs = np.concatenate([s.high for s in spaces], axis=0)
        return Box(lows, highs)


# GroundTruthEncoder returns the ground truth observation as a single vector
class GroundTruthEncoder(ObservationEncoder):
    requires_vision = False # whether the encoder requires vision (rendering) or not
    latent_encoding = False

    def encode(self, obs):
        obs_list = [obs[key] for key in self.obs_keys]
        if len(obs_list) > 0:
            return np.concatenate(obs_list, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32)

    def get_space(self, robo_env):
        o = robo_env.observation_spec()
        dim = sum([o[key].shape[0] for key in self.obs_keys])
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(dim,))



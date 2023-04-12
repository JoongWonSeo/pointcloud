import numpy as np
from abc import ABC, abstractmethod

class Sensor(ABC):
    '''
    Abstract base class for a sensor module
    It is essentially a layer between the environment and the encoder,
    converting the state into a observation that can be encoded later

    Inheriting classes must implement the following:
        - observe(state): given the ground truth state, returns the observation dict

    Inheriting classes may implement the following (optional):
        - reset(): called when the environment is reset
        - env_kwargs: kwargs when initializing robosuite env, e.g. camera settings
    '''
    requires_vision = False

    def __init__(self, env):
        self.env = env
    
    @property
    def env_kwargs(self):
        return {}
    
    def reset(self):
        pass
    
    @abstractmethod
    def observe(self, state):
        '''
        Returns a observation dict for the given ground truth state
        '''
        pass


class PassthroughSensor(Sensor):
    requires_vision = False

    def observe(self, state):
        return state

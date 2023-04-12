from gymnasium.envs.registration import register
from .envs import RoboReach, RoboPush, RoboPickAndPlace

register(
    id='RoboReach-v0',
    entry_point=RoboReach,
    max_episode_steps=50,
)

register(
    id='RoboPush-v0',
    entry_point=RoboPush,
    max_episode_steps=50,
)

register(
    id='RoboPickAndPlace-v0',
    entry_point=RoboPickAndPlace,
    max_episode_steps=50,
)


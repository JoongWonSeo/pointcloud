from gymnasium.envs.registration import register
from .envs import RobosuiteReach, RobosuitePush, RobosuitePickAndPlace

register(
    id='RobosuiteReach-v0',
    entry_point=RobosuiteReach,
    max_episode_steps=50,
)

register(
    id='RobosuitePush-v0',
    entry_point=RobosuitePush,
    max_episode_steps=50,
)

register(
    id='RobosuitePickAndPlace-v0',
    entry_point=RobosuitePickAndPlace,
    max_episode_steps=50,
)


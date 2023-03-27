from gymnasium.envs.registration import register
from .envs import RobosuiteReach, RobosuiteLift, RobosuitePickAndPlace, RobosuiteDoor, RobosuitePeg

register(
    id='RobosuiteReach-v0',
    entry_point=RobosuiteReach,
    max_episode_steps=50,
)

register(
    id='RobosuiteLift-v0',
    entry_point=RobosuiteLift,
    max_episode_steps=50,
)

register(
    id='RobosuitePickAndPlace-v0',
    entry_point=RobosuitePickAndPlace,
    max_episode_steps=50,
)

register(
    id='RobosuiteDoor-v0',
    entry_point=RobosuiteDoor,
    max_episode_steps=50,
)

register(
    id='RobosuitePeg-v0',
    entry_point=RobosuitePeg,
    max_episode_steps=50,
)
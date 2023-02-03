from gymnasium.envs.registration import register
from .robosuite import RobosuiteReach

register(
    id='RobosuiteReach-v0',
    entry_point='gym_robosuite_envs.robosuite:RobosuiteReach',
    max_episode_steps=100,
)

register(
    id='RobosuiteLift-v0',
    entry_point='gym_robosuite_envs.robosuite:RobosuiteLift',
    max_episode_steps=100,
)

register(
    id='RobosuitePickAndPlace-v0',
    entry_point='gym_robosuite_envs.robosuite:RobosuitePickAndPlace',
    max_episode_steps=100,
)
from gymnasium.envs.registration import register
from .robosuite import RobosuiteReach

register(
    id='RobosuiteReach-v0',
    entry_point='gym_robosuite_envs.robosuite:RobosuiteReach',
    max_episode_steps=100,
)
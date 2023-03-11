from gymnasium.envs.registration import register
from .robosuite import RobosuiteReach, RobosuiteLift, RobosuitePickAndPlace
from .pc_encoder import PointCloudGTPredictor

register(
    id='RobosuiteReach-v0',
    entry_point=RobosuiteReach,
    max_episode_steps=100,
)

register(
    id='RobosuiteLift-v0',
    entry_point=RobosuiteLift,
    max_episode_steps=100,
)

register(
    id='RobosuitePickAndPlace-v0',
    entry_point=RobosuitePickAndPlace,
    max_episode_steps=100,
)
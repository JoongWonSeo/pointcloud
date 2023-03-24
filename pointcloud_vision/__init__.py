from .pc_encoder import PointCloudSensor, PointCloudGTPredictor

from gymnasium.envs.registration import register
from robosuite_envs.envs import RobosuiteReach, RobosuiteLift, RobosuitePickAndPlace


register(
    id='VisionLift-v0',
    entry_point=RobosuiteLift,
    max_episode_steps=100,
    kwargs={
        'sensor':PointCloudSensor,
        'obs_encoder':PointCloudGTPredictor,
        'goal_encoder':PointCloudGTPredictor,
    }
)

register(
    id='VisionPickAndPlace-v0',
    entry_point=RobosuitePickAndPlace,
    max_episode_steps=100,
    kwargs={
        'sensor':PointCloudSensor,
        'obs_encoder':PointCloudGTPredictor,
        'goal_encoder':PointCloudGTPredictor,
    }
)
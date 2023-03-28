from .pc_encoder import PointCloudSensor, PointCloudGTPredictor, PointCloudEncoder

from gymnasium.envs.registration import register
from robosuite_envs.envs import RobosuiteReach, RobosuiteLift, RobosuitePickAndPlace

register(
    id='VisionReach-v0',
    entry_point=RobosuiteReach,
    max_episode_steps=50,
    kwargs={
        'sensor':PointCloudSensor,
        'obs_encoder':PointCloudEncoder,
        'goal_encoder':PointCloudEncoder,
    }
)

register(
    id='VisionLift-v0',
    entry_point=RobosuiteLift,
    max_episode_steps=50,
    kwargs={
        'sensor':PointCloudSensor,
        'obs_encoder':PointCloudGTPredictor,
        'goal_encoder':PointCloudGTPredictor,
    }
)

register(
    id='VisionPickAndPlace-v0',
    entry_point=RobosuitePickAndPlace,
    max_episode_steps=50,
    kwargs={
        'sensor':PointCloudSensor,
        'obs_encoder':PointCloudGTPredictor,
        'goal_encoder':PointCloudGTPredictor,
    }
)
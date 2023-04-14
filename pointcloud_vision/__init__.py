from .pc_sensor import PointCloudSensor
from .pc_encoder import GlobalAEEncoder, GlobalSegmenterEncoder, MultiSegmenterEncoder, StatePredictor, StatePredictorVisualGoal

from gymnasium.envs.registration import register
from robosuite_envs.envs import RoboReach, RoboPush, RoboPickAndPlace

register(
    id='VisionReach-v0',
    entry_point=RoboReach,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': GlobalAEEncoder,
    }
)

register(
    id='VisionReachMultiSeg-v0',
    entry_point=RoboReach,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': MultiSegmenterEncoder,
        'simulate_goal': True,
    }
)

# register(
#     id='VisionReachGT-v0',
#     entry_point=RoboReach,
#     max_episode_steps=50,
#     kwargs={
#         'sensor': PointCloudSensor,
#         'encoder': GlobalSegmenterEncoder,
#     }
# )

register(
    id='VisionPush-v0',
    entry_point=RoboPush,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': MultiSegmenterEncoder,
    }
)

register(
    id='VisionPushSeg-v0',
    entry_point=RoboPush,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': GlobalSegmenterEncoder,
    }
)

register(
    id='VisionPushMultiSeg-v0',
    entry_point=RoboPush,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': MultiSegmenterEncoder,
    }
)

register(
    id='VisionPushGT-v0',
    entry_point=RoboPush,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': StatePredictor,
    }
)

# register(
#     id='VisionPush-v0',
#     entry_point=RoboPush,
#     max_episode_steps=50,
#     kwargs={
#         'sensor':PointCloudSensor,
#         'obs_encoder':PointCloudGTPredictor,
#         'goal_encoder':PointCloudGTPredictor,
#     }
# )

register(
    id='VisionPickAndPlace-v0',
    entry_point=RoboPickAndPlace,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': StatePredictor,
    }
)

register(
    id='VisionPickAndPlaceMultiSeg-v0',
    entry_point=RoboPickAndPlace,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': MultiSegmenterEncoder,
    }
)

register(
    id='VisionPickAndPlaceGT-v0',
    entry_point=RoboPickAndPlace,
    max_episode_steps=50,
    kwargs={
        'sensor': PointCloudSensor,
        'encoder': StatePredictor,
    }
)
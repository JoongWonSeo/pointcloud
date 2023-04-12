from .pc_encoder import PointCloudSensor, PointCloudGTPredictor, GlobalAEEncoder, GlobalSegmenterEncoder

from gymnasium.envs.registration import register
from robosuite_envs.envs import RoboReach, RoboPush, RoboPickAndPlace

register(
    id='VisionReach-v0',
    entry_point=RoboReach,
    max_episode_steps=50,
    kwargs={
        'sensor':PointCloudSensor,
        'encoder':GlobalAEEncoder,
    }
)

# register(
#     id='GTReach-v0',
#     entry_point=RoboReach,
#     max_episode_steps=50,
#     kwargs={
#         'sensor':PointCloudSensor,
#         'obs_encoder':PointCloudGTPredictor,
#     }
# )

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

# register(
#     id='VisionPickAndPlace-v0',
#     entry_point=RoboPickAndPlace,
#     max_episode_steps=50,
#     kwargs={
#         'sensor':PointCloudSensor,
#         'obs_encoder':PointCloudGTPredictor,
#         'goal_encoder':PointCloudGTPredictor,
#     }
# )
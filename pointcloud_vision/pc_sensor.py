import torch
from torchvision.transforms import Compose
from robosuite_envs.sensors import Sensor
from pointcloud_vision.utils import FilterBBox, SampleRandomPoints, SampleFurthestPoints, Normalize, Unnormalize, obs_to_pc
from robosuite_envs.utils import multiview_pointcloud #TODO move to vision module
import pointcloud_vision.cfg as cfg



class PointCloudSensor(Sensor):
    '''
    Sensor that generates a pointcloud from a 2.5D observation
    observe returns a dict that is compatible with the PointCloudDataset save format,
    including keys 'points' and features like 'rgb', 'segmentation', and also 'boundingbox' and 'classes'
    '''
    requires_vision = True

    def __init__(self, env, require_segmentation=False):
        super().__init__(env)

        self.features = ['rgb'] + (['segmentation'] if require_segmentation else [])
        self.bbox = torch.as_tensor(env.bbox).to(cfg.device)
        sampler = {'FPS': SampleFurthestPoints, 'RS': SampleRandomPoints, None: lambda x: (lambda y: y)}[env.sampler]
        self.preprocess = Compose([
            FilterBBox(self.bbox),
            sampler(env.sample_points),
        ])

    @property
    def env_kwargs(self):
        return super().env_kwargs | {
            'camera_depths': True,
            'camera_segmentations': 'instance' if 'segmentation' in self.features else None,
        }
    
    def observe(self, state):
        '''
        Even though this depends on the self.env.robo_env.sim, it only uses it to get the camera poses, so *as long as the camera poses are the same*, the generated pointclouds will only depend on the given state, even for a different environment (so you can use this for the goal imagination env).
        '''
        # generate pointcloud from 2.5D observations
        pc, feats = multiview_pointcloud(self.env.robo_env.sim, state, self.env.cameras, self.preprocess, self.features, cfg.device)
        # TODO: currently, the original state is also included in the observation, in order to allow GT Encoders to work as well.
        return state | {'points': pc, 'boundingbox': self.bbox} | feats

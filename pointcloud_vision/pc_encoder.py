import os
import pointcloud_vision.cfg as cfg
import torch
import numpy as np
from robosuite_envs.encoders import ObservationEncoder
from robosuite_envs.sensors import Sensor
from robosuite_envs.utils import multiview_pointcloud #TODO move to vision module
from pointcloud_vision.models.pc_encoders import backbone_factory, GTEncoder
import pointcloud_vision.train
from pointcloud_vision.utils import FilterBBox, SampleFurthestPoints, Normalize, Unnormalize, obs_to_pc
from torchvision.transforms import Compose
from gymnasium.spaces import Box

class PointCloudSensor(Sensor):
    '''
    Sensor that generates a pointcloud from a 2.5D observation
    observe returns a dict that is compatible with the PointCloudDataset save format,
    including keys 'points' and features like 'rgb', 'segmentation', and also 'boundingbox' and 'classes'
    '''

    requires_vision = True

    def __init__(self, env):
        super().__init__(env)

        self.features = ['rgb', 'segmentation']
        self.bbox = torch.as_tensor(env.bbox).to(cfg.device)
        self.preprocess = Compose([
            FilterBBox(self.bbox),
            SampleFurthestPoints(self.env.sample_points),
        ])

    @property
    def env_kwargs(self):
        return super().env_kwargs | {
            'camera_depths': True,
            'camera_segmentations': 'class',
        }
    
    def observe(self, state):
        '''
        Even though this depends on the self.env.robo_env.sim, it only uses it to get the camera poses, so *as long as the camera poses are the same*, the generated pointclouds will only depend on the given state, even for a different environment.
        '''

        # generate pointcloud from 2.5D observations
        pc, feats = multiview_pointcloud(self.env.robo_env.sim, state, self.env.cameras, self.preprocess, self.features, cfg.device)
        # TODO: currently, the original state is also included in the observation, in order to allow GT Encoders to work as well.
        return state | {'points': pc, 'boundingbox': self.bbox} | feats


class PointCloudGTPredictor(ObservationEncoder):
    '''
    '''
    requires_vision = True
    latent_encoding = False

    # configure ground-truth data pre/postprocessing for each environment
    cfgs = {}
    cfgs['Lift'] = {
        'to_gt': lambda bbox: Unnormalize(bbox), # unnormalize cube position
        'from_gt': lambda bbox: Normalize(bbox), # normalize cube position
    }

    def __init__(self, env, obs_keys):
        super().__init__(env, obs_keys)
        
        if self.obs_keys == ['cube_pos']:
            # Cube position predictor from pointcloud (Point Cloud {XYZRGB} -> Cube (XYZ))
            self.features = ['rgb']
            feature_dims = 3
            self.encoding_dim = env.gt_dim
            self.postprocess_fn = self.cfgs['Lift']['to_gt']

            load_dir = os.path.join(os.path.dirname(__file__), 'output/Lift/GTEncoder_PointNet2/version_1/checkpoints/epoch=99-step=2000.ckpt')
            self.pc_encoder = GTEncoder(backbone_factory['PointNet2'](feature_dims=feature_dims), self.encoding_dim)
            self.pc_encoder = pointcloud_vision.train.Lit(self.pc_encoder, None)
            self.pc_encoder.load_state_dict(torch.load(load_dir)['state_dict'])

        else:
            raise NotImplementedError()

        self.pc_encoder = self.pc_encoder.model.to(cfg.device)
        self.pc_encoder.eval()

    def encode(self, obs):
        preprocess = Normalize(obs['boundingbox'])
        postprocess = self.postprocess_fn(obs['boundingbox'])
    
        pc = preprocess(obs_to_pc(obs, self.features)).unsqueeze(0)
        pred = postprocess(self.pc_encoder(pc).detach()).squeeze(0)

        return pred.cpu().numpy()
    
    def get_space(self, robo_env):
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.encoding_dim,))


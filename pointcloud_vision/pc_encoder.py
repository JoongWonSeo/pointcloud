import os
import pointcloud_vision.cfg as cfg
import torch
import numpy as np
from robosuite_envs.encoders import ObservationEncoder
from robosuite_envs.sensors import Sensor
from robosuite_envs.utils import multiview_pointcloud #TODO move to vision module
from pointcloud_vision.models.pc_encoders import backbone_factory, GTEncoder
from pointcloud_vision.train import Lit
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
        self.bbox = torch.Tensor(env.bbox).to(cfg.device)
        self.preprocess = Compose([
            FilterBBox(self.bbox),
            SampleFurthestPoints(self.env.sample_points),
            Normalize(self.bbox),
        ])

    @property
    def env_kwargs(self):
        return super().env_kwargs | {
            'camera_depths': True,
            'camera_segmentations': 'class',
        }
    
    def observe(self, state):
        # generate pointcloud from 2.5D observations
        pc, feats = multiview_pointcloud(self.env.robo_env.sim, state, self.env.cameras, self.preprocess, self.features, cfg.device)
        # TODO: currently, the original state is also included in the observation, in order to allow GT Encoders to work as well.
        return state | {'points': pc, 'boundingbox': self.bbox} | feats

class PointCloudGTPredictor(ObservationEncoder):
    '''
    '''
    latent_encoding = False

    def __init__(self, env, obs_keys):
        super().__init__(env, obs_keys)
        
        if self.obs_keys == ['cube_pos']:
            # Cube position predictor from pointcloud (Point Cloud {XYZRGB} -> Cube (XYZ))
            self.features = ['rgb']
            feature_dims = 3
            self.encoding_dim = 3

            # load_dir = '../pointcloud_vision/output/Lift/GTEncoder_PointNet2/version_0/checkpoints/epoch=99-step=2000.ckpt'
            load_dir = os.path.join(os.path.dirname(__file__), 'output/Lift/GTEncoder_PointNet2/version_0/checkpoints/epoch=99-step=2000.ckpt')
            self.pc_encoder = GTEncoder(backbone_factory['PointNet2'](feature_dims=feature_dims), self.encoding_dim)
            self.pc_encoder = Lit(self.pc_encoder, None)
            self.pc_encoder.load_state_dict(torch.load(load_dir)['state_dict'])
            # self.pc_encoder, _ = create_model('GTEncoder', 'PointNet2', load_dir=)

        else:
            raise NotImplementedError()

        self.pc_encoder = self.pc_encoder.model.to(cfg.device)
        self.pc_encoder.eval()

    def encode(self, obs):
        pc = obs_to_pc(obs, self.features).unsqueeze(0)
        pred = self.pc_encoder(pc).detach()

        # unnormalize to world coordinates
        pred = Unnormalize(obs['boundingbox'])(pred)

        return pred.squeeze(0).cpu().numpy()
    
    def get_space(self, robo_env):
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.encoding_dim,))


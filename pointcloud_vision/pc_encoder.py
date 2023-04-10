import os
import pointcloud_vision.cfg as cfg
import torch
import numpy as np
from robosuite_envs.encoders import ObservationEncoder
from robosuite_envs.sensors import Sensor
from robosuite_envs.utils import multiview_pointcloud #TODO move to vision module
from pointcloud_vision.models.pc_encoders import backbone_factory, AE, GTEncoder
from pointcloud_vision.train import create_model
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



class PointCloudEncoder(ObservationEncoder):
    requires_vision = True
    latent_encoding = True

    def __init__(self, env, obs_keys):
        super().__init__(env, obs_keys)
        
        if self.obs_keys == ['robot0_eef_pos']:
            # use the encoder trained from Reach task
            self.features = ['rgb']
            self.encoding_dim = cfg.bottleneck_size

            load_dir = os.path.join(os.path.dirname(__file__), 'output/Reach/Autoencoder_PointNet2/version_3/checkpoints/epoch=99-step=8000.ckpt')
            lit, _ = create_model('Autoencoder', 'PointNet2', env, load_dir)
            # TODO LOAD PRECALIBRATED LATENT THRESHOLD

            self.pc_encoder = lit.model.encoder.to(cfg.device)
            

        elif self.obs_keys == ['cube_pos']:
            raise NotImplementedError() #TODO from lift dataset
        else:
            raise NotImplementedError()
        
        self.pc_encoder.eval()
    
    def encode(self, obs):
        preprocess = Normalize(obs['boundingbox'])
    
        pc = preprocess(obs_to_pc(obs, self.features)).unsqueeze(0)
        pred = self.pc_encoder(pc).detach().squeeze(0)

        return pred.cpu().numpy()
    
    def get_space(self, robo_env):
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.encoding_dim,))


        
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
    cfgs['Reach'] = {
        'to_gt': lambda bbox: Unnormalize(bbox), # unnormalize eef position
        'from_gt': lambda bbox: Normalize(bbox), # normalize eef position
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
            lit, _ = create_model('GTEncoder', 'PointNet2', env, load_dir)
            self.pc_encoder = lit.model.to(cfg.device)

        elif self.obs_keys == ['robot0_eef_pos']:
            # EEF position predictor from pointcloud (Point Cloud {XYZRGB} -> EEF (XYZ))
            self.features = ['rgb']
            feature_dims = 3
            self.encoding_dim = env.gt_dim
            self.postprocess_fn = self.cfgs['Reach']['to_gt']

            load_dir = os.path.join(os.path.dirname(__file__), 'output/Reach/GTEncoder_PointNet2/version_1/checkpoints/epoch=99-step=8000.ckpt')
            lit, _ = create_model('GTEncoder', 'PointNet2', env, load_dir)
            self.pc_encoder = lit.model.to(cfg.device)
            
        else:
            raise NotImplementedError()

        self.pc_encoder.eval()

    def encode(self, obs):
        preprocess = Normalize(obs['boundingbox'])
        postprocess = self.postprocess_fn(obs['boundingbox'])
    
        pc = preprocess(obs_to_pc(obs, self.features)).unsqueeze(0)
        pred = postprocess(self.pc_encoder(pc).detach()).squeeze(0)

        return pred.cpu().numpy()
    
    def get_space(self, robo_env):
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.encoding_dim,))

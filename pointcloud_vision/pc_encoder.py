import os
import pointcloud_vision.cfg as cfg
import torch
import numpy as np
from robosuite_envs.encoders import ObservationEncoder
from robosuite_envs.sensors import Sensor
from robosuite_envs.utils import multiview_pointcloud #TODO move to vision module
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
        Even though this depends on the self.env.robo_env.sim, it only uses it to get the camera poses, so *as long as the camera poses are the same*, the generated pointclouds will only depend on the given state, even for a different environment (so you can use this for the goal imagination env).
        '''
        # generate pointcloud from 2.5D observations
        pc, feats = multiview_pointcloud(self.env.robo_env.sim, state, self.env.cameras, self.preprocess, self.features, cfg.device)
        # TODO: currently, the original state is also included in the observation, in order to allow GT Encoders to work as well.
        return state | {'points': pc, 'boundingbox': self.bbox} | feats


####### Utils #######

def model_path(scene, model, backbone='PointNet2', version=None):
    if version is not None:
        model_dir = f'output/{scene}/{model}_{backbone}/version_{version}/checkpoints/'
        model_dir = os.path.join(os.path.dirname(__file__), model_dir)
    else:
        model_dir = f'output/{scene}/{model}_{backbone}/'
        model_dir = os.path.join(os.path.dirname(__file__), model_dir)
        model_dir += sorted(map(lambda n: (len(n), n), os.listdir(model_dir)))[-1][1] # lastest version, sorted first by length and then by name
        model_dir += '/checkpoints/'
    
    model_dir += sorted(os.listdir(model_dir))[-1] # lastest checkpoint
    return model_dir

def load_model(scene, model, backbone, version=None):
    load_dir = model_path(scene, model, backbone, version)
    lit, _ = create_model(model, backbone, scene, load_dir)
    return lit.model



class GlobalSceneEncoder(ObservationEncoder):
    '''
    For the typical Encoder-Decoder architecture with a single global latent vector
    The entire scene is encoded into a single latent vector
    Thus, encoders inherently require observation space to be the same as goal space
    Or rather, the keys simply do not matter
    i.e. Autoencoder, Segmenter, but not MultiDecoder or GTEncoder
    '''
    requires_vision = True
    latent_encoding = True

    def __init__(self, env, obs_keys, goal_keys, model, backbone, version=None):
        super().__init__(env, obs_keys, goal_keys)

        # sanity checks
        if model not in ['Autoencoder', 'Segmenter']:
            raise NotImplementedError()
        
        self.features = ['rgb']
        self.encoding_dim = sum(env.class_latent_dim)

        model = load_model(env.scene, model, backbone, version)
        # TODO LOAD PRECALIBRATED LATENT THRESHOLD
        # self.latent_threshold = model.latent_threshold

        self.pc_encoder = model.encoder.to(cfg.device).eval()
    
    def encode_observation(self, obs):
        preprocess = Normalize(obs['boundingbox'])
    
        pc = preprocess(obs_to_pc(obs, self.features)).unsqueeze(0)
        pred = self.pc_encoder(pc).detach().squeeze(0)

        return pred.cpu().numpy()
    
    def encode_goal(self, obs):
        return self.encode_observation(obs)
    
    def __call__(self, obs):
        enc = self.encode_observation(obs)
        return enc, enc
    
    def get_encoding_space(self, robo_env):
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.encoding_dim,))
    
    def get_goal_space(self, robo_env):
        return self.get_encoding_space(robo_env)


# Specific Encoders
def GlobalAEEncoder(env, obs_keys, goal_keys):
    return GlobalSceneEncoder(env, obs_keys, goal_keys, 'Autoencoder', 'PointNet2')

def GlobalSegmenterEncoder(env, obs_keys, goal_keys):
    return GlobalSceneEncoder(env, obs_keys, goal_keys, 'Segmenter', 'PointNet2')



# TODO: this should be replaced by a multi-bottle archietecture
class PointCloudGTPredictor(ObservationEncoder):
    '''
    '''
    requires_vision = True
    latent_encoding = False

    # configure ground-truth data pre/postprocessing for each scene
    cfgs = {}
    cfgs['Table'] = {
        'to_gt': lambda bbox: Unnormalize(bbox), # unnormalize eef position
        'from_gt': lambda bbox: Normalize(bbox), # normalize eef position
    }
    cfgs['Cube'] = {
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

            dataset = 'Lift'
            model = 'GTEncoder'
            backbone = 'PointNet2'

            model = load_model(env, dataset, model, backbone)
            self.pc_encoder = model.encoder.to(cfg.device)

        elif self.obs_keys == ['robot0_eef_pos']:
            # EEF position predictor from pointcloud (Point Cloud {XYZRGB} -> EEF (XYZ))
            self.features = ['rgb']
            feature_dims = 3
            self.encoding_dim = env.gt_dim
            self.postprocess_fn = self.cfgs['Reach']['to_gt']

            dataset = 'Reach'
            model = 'GTEncoder'
            backbone = 'PointNet2'

            model = load_model(env, dataset, model, backbone)
            self.pc_encoder = model.encoder.to(cfg.device)
            
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

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

def save_metadata(data_dict, scene, model, backbone, version=None):
    '''Use this to save any data alongside the model!'''
    file = model_path(scene, model, backbone, version)
    file = file.replace('/checkpoints/', '/metadata/').replace('.ckpt', '.npz')
    # first create directory if it doesn't exist
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.savez(file, **data_dict)
    return file

def load_metadata(scene, model, backbone, version=None):
    file = model_path(scene, model, backbone, version)
    file = file.replace('/checkpoints/', '/metadata/').replace('.ckpt', '.npz')
    data = np.load(file)
    return data



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

        self.env = env
        self.model = model
        self.backbone = backbone
        self.version = version

        # sanity checks
        if model not in ['Autoencoder', 'Segmenter']:
            raise NotImplementedError()
        
        self.features = ['rgb']
        self.encoding_dim = sum(env.class_latent_dim)

        model = load_model(env.scene, model, backbone, version)
        self.latent_threshold = self.load_latent_threshold()
        
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
    
    def load_latent_threshold(self):
        try:
            data = load_metadata(self.env.scene, self.model, self.backbone, self.version)
            return data['latent_threshold']
        except:
            print('No latent threshold found! Make sure to calibrate the encoder!')
            return None
    
    def save_latent_threshold(self, threshold):
        save_metadata({'latent_threshold': threshold}, self.env.scene, self.model, self.backbone, self.version)
        self.latent_threshold = threshold


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

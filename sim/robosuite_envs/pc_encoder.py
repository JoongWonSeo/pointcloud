import cfg
import torch
import numpy as np
from robosuite.utils.camera_utils import CameraMover, get_real_depth_map
from robosuite.utils import transform_utils
from .base import ObservationEncoder
from vision.models.pn_autoencoder import PN2PosExtractor
from sim.utils import to_pointcloud, multiview_pointcloud
from vision.utils import FilterBBox, SampleFurthestPoints, Normalize
from torchvision.transforms import Compose
from gymnasium.spaces import Box


class PointCloudEncoder(ObservationEncoder):
    def __init__(self, proprioception_keys, cameras, camera_poses, camera_size=cfg.camera_size, bbox=cfg.bbox, sample_points=2048, robo_env=None):
        super().__init__(proprioception_keys, robo_env) #TODO add to init args

        self.cameras = cameras
        self.camera_poses = camera_poses
        self.bbox = torch.Tensor(bbox) # 3D bounding box [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.sample_points = sample_points

        self.env_kwargs = {
            'use_camera_obs': True,
            'camera_names': self.cameras,
            'camera_widths': camera_size[0],
            'camera_heights': camera_size[1],
            'camera_depths': True,
            'camera_segmentations': 'class',
        }
        
        self.pc_encoder = PN2PosExtractor(3) # segmenting autoencoder
        self.pc_encoder.load_state_dict(torch.load('../vision/weights/PC_PP.pth'))
        self.pc_encoder = self.pc_encoder.to('cuda')
        self.pc_encoder.eval()

        self.transform = Compose([
            FilterBBox(bbox),
            SampleFurthestPoints(2048),
            Normalize(bbox)
        ])
        
    def reset(self, obs):
        # setup cameras and camera poses
        self.camera_movers = [CameraMover(self.robo_env, camera=c) for c in self.cameras]
        for mover, pose in zip(self.camera_movers, self.camera_poses):
            mover.set_camera_pose(pose)
        
        return self.encode(obs)
    
    def encode_state(self, obs):
        # generate pointcloud from 2.5D observations
        pc, feats = multiview_pointcloud(self.robo_env.sim, obs, self.cameras, self.transform)
        
        # encode pointcloud
        pc = pc.unsqueeze(0).to('cuda')
        embedding = self.pc_encoder(pc)

        return embedding.squeeze(0).detach().cpu().numpy()
    
    def get_space(self):
        o = self.robo_env.observation_spec()
        dim = 3 + sum([o[key].shape[0] for key in self.proprioception_keys])
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(dim,))



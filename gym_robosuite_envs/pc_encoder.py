import torch
import numpy as np
from robosuite.utils.camera_utils import CameraMover, get_real_depth_map
from .base import ObservationEncoder
from models.pn_autoencoder import PNAutoencoder
from utils import to_pointcloud, filter_pointcloud
from train_utils import SampleFurthestPoints, Normalize
from torchvision.transforms import Compose
from gymnasium.spaces import Box


class PointCloudEncoder(ObservationEncoder):
    def __init__(self, cameras, camera_poses, bbox, sample_points=2048, robo_env=None):
        self.cameras = cameras
        self.camera_poses = camera_poses
        self.bbox = torch.Tensor(bbox) # 3D bounding box [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.sample_points = sample_points
        self.robo_env = robo_env # this can be overwritten by GoalEnvRobosuite in the constructor
        
        self.pc_encoder = PNAutoencoder(out_points=sample_points, dim_per_point=6)
        self.pc_encoder.load_state_dict(torch.load('weights/PC_AE.pth'))
        self.pc_encoder = self.pc_encoder.encoder.to('cuda')
        self.pc_encoder.eval()

        self.transform = Compose([
            SampleFurthestPoints(2048),
            Normalize([[-0.5, 0.5], [-0.5, 0.5], [0.5, 1.5]]),
        ])
        
    def reset(self, obs):
        # setup cameras and camera poses
        self.camera_movers = [CameraMover(self.robo_env, camera=c) for c in self.cameras]
        for mover, pose in zip(self.camera_movers, self.camera_poses):
            mover.set_camera_pose(pose)
        
        return self.encode(obs)
    
    def encode_state(self, obs):
        # generate pointcloud from 2.5D observations
        pcs, rgbs = [], []
        for c in self.cameras:
            img = obs[c + '_image'] / 255
            depth_map = get_real_depth_map(self.robo_env.sim, c + '_depth')

            pc, rgb = to_pointcloud(self.robo_env.sim, img, depth_map, c)
            pcs.append(pc)
            rgbs.append(rgb)
        
        pcs = np.concatenate(pcs, axis=0)
        rgbs = np.concatenate(rgbs, axis=0)

        # filter points outside of bounding box
        pcs, rgbs = filter_pointcloud(pcs, rgbs, self.bbox)

        # sample points
        pc = torch.tensor(np.hstack((pcs, rgbs)).astype(np.float32))
        pc = self.transform(pc)

        # encode pointcloud
        pc = pc.unsqueeze(0).to('cuda')
        embedding = self.pc_encoder(pc)

        return embedding.squeeze(0).cpu().numpy()
    
    def get_space(self):
        o = self.robo_env.observation_spec()
        dim = self.pc_encoder.out_channels + sum([o[key].shape[0] for key in self.proprioception_keys])
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(dim,))



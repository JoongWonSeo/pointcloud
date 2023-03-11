import cfg
import torch
import numpy as np
from robosuite.utils.camera_utils import CameraMover, get_real_depth_map
from robosuite.utils import transform_utils
from .base import ObservationEncoder
from vision.train import create_model
from sim.utils import to_pointcloud, multiview_pointcloud
from vision.utils import FilterBBox, SampleFurthestPoints, Normalize, Unnormalize
from torchvision.transforms import Compose
from gymnasium.spaces import Box


class PointCloudGTPredictor(ObservationEncoder):
    def __init__(self, proprioception_keys, cameras=list(cfg.camera_poses.keys()), camera_poses=list(cfg.camera_poses.values()), camera_size=cfg.camera_size, bbox=cfg.bbox, sample_points=cfg.pc_sample_points, robo_env=None):
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
        
        # self.pc_encoder = PN2PosExtractor(6) # PC[XYZRGB] -> Cube[XYZ]
        # self.pc_encoder.load_state_dict(torch.load('../vision/weights/PC_PP_RGB.pth')['model'])
        # self.pc_encoder = Lit.load_from_checkpoint('../vision/weights/PC_PP_RGB.ckpt', predictor=PN2PosExtractor(6), loss_fn=None).model
        self.pc_encoder, _ = create_model('GTEncoder', 'PointNet2', load_dir='../vision/output/Lift/GTEncoder_PointNet2/version_0/checkpoints/epoch=99-step=2000.ckpt')
        self.pc_encoder = self.pc_encoder.model.to(cfg.device)
        self.pc_encoder.eval()

        self.preprocess = Compose([
            FilterBBox(bbox),
            SampleFurthestPoints(sample_points),
            Normalize(bbox)
        ])
        self.postprocess = Unnormalize(bbox)
        
    def reset(self, obs):
        # setup cameras and camera poses
        self.camera_movers = [CameraMover(self.robo_env, camera=c) for c in self.cameras]
        for mover, pose in zip(self.camera_movers, self.camera_poses):
            mover.set_camera_pose(np.array(pose[0]), np.array(pose[1]))
        
        return self.encode(obs) #TODO: due to cameramovers, the actual is no longer same
    
    def encode_state(self, obs):
        # generate pointcloud from 2.5D observations
        pc, feats = multiview_pointcloud(self.robo_env.sim, obs, self.cameras, self.preprocess, ['rgb'])
        pc = torch.cat((pc, feats['rgb']), dim=1)
        
        # encode pointcloud
        pc = pc.unsqueeze(0).to(cfg.device)
        pred = self.pc_encoder(pc).detach().cpu()

        # unnormalize to world coordinates
        pred = self.postprocess(pred)

        return pred.squeeze(0).numpy()
    
    def get_space(self):
        o = self.robo_env.observation_spec()
        dim = 3 + sum([o[key].shape[0] for key in self.proprioception_keys])
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(dim,))


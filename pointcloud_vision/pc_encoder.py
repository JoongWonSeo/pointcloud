import pointcloud_vision.cfg as cfg
import torch
import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map
from robosuite.utils import transform_utils
from robosuite_envs.base import ObservationEncoder
from pointcloud_vision.train import create_model
from robosuite_envs.utils import to_pointcloud, multiview_pointcloud
from pointcloud_vision.utils import FilterBBox, SampleFurthestPoints, Normalize, Unnormalize
from torchvision.transforms import Compose
from gymnasium.spaces import Box


class PointCloudGTPredictor(ObservationEncoder):
    '''
    Cube position predictor from pointcloud
    Point Cloud {XYZRGB} -> Cube (XYZ)
    '''
    def __init__(self, proprioception_keys, cameras=cfg.camera_poses, camera_size=cfg.camera_size, bbox=cfg.bbox, sample_points=cfg.pc_sample_points, robo_env=None):
        super().__init__(proprioception_keys, cameras, camera_size, robo_env) #TODO add to init args

        self.bbox = torch.Tensor(bbox) # 3D bounding box [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.sample_points = sample_points
        
        # self.pc_encoder = PN2PosExtractor(6) # PC[XYZRGB] -> Cube[XYZ]
        # self.pc_encoder.load_state_dict(torch.load('../vision/weights/PC_PP_RGB.pth')['model'])
        # self.pc_encoder = Lit.load_from_checkpoint('../vision/weights/PC_PP_RGB.ckpt', predictor=PN2PosExtractor(6), loss_fn=None).model
        self.pc_encoder, _ = create_model('GTEncoder', 'PointNet2', load_dir='../pointcloud_vision/output/Lift/GTEncoder_PointNet2/version_0/checkpoints/epoch=99-step=2000.ckpt')
        self.pc_encoder = self.pc_encoder.model.to(cfg.device)
        self.pc_encoder.eval()

        self.preprocess = Compose([
            FilterBBox(bbox),
            SampleFurthestPoints(sample_points),
            Normalize(bbox)
        ])
        self.postprocess = Unnormalize(bbox)
    
    @property
    def env_kwargs(self):
        return super().env_kwargs | {
            'camera_depths': True,
            'camera_segmentations': 'class',
        }
    
    def encode_state(self, obs):
        return np.array([], dtype=np.float32)

    def encode_goal(self, obs):
        # generate pointcloud from 2.5D observations
        pc, feats = multiview_pointcloud(self.robo_env.sim, obs, self.cameras, self.preprocess, ['rgb'])
        pc = torch.cat((pc, feats['rgb']), dim=1)
        
        # encode pointcloud (predict cube position)
        pc = pc.unsqueeze(0).to(cfg.device)
        pred = self.pc_encoder(pc).detach().cpu()

        # unnormalize to world coordinates
        pred = self.postprocess(pred)

        return pred.squeeze(0).numpy()
    
    def get_space(self):
        o = self.robo_env.observation_spec()
        dim = 3 + sum([o[key].shape[0] for key in self.proprioception_keys])
        return Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(dim,))


import torch
import torch.nn
from pytorch3d.ops import sample_farthest_points
from pytorch3d import loss as pytorch3d_loss
from loss.emd.emd_module import emdModule



# Point Cloud Transformations

class SampleFurthestPoints:
    def __init__(self, K):
        self.K = K
    
    def __call__(self, points):
        # convert to batch of 1 pointcloud
        points = points.unsqueeze(0) # (1, N, D)
        # sample K points
        points, _ = sample_farthest_points(points, K=self.K)
        # convert back to (N, D)
        points = points.squeeze(0) # (N, D)

        return points

class Normalize:
    def __init__(self, bbox):
        '''
        bbox: 3D bounding box of the point cloud (x_min, x_max, y_min, y_max, z_min, z_max)
        '''
        self.min = torch.Tensor(bbox[0:6:2])
        self.max = torch.Tensor(bbox[1:6:2])

    def __call__(self, points):
        # normalize points along each axis (only the first 3 dimensions)
        points[:, 0:3] = (points[:, 0:3] - self.min) / (self.max - self.min)
        return points


# Loss Functions

class chamfer_distance:
    def __init__(self):
        self.loss_fn = pytorch3d_loss.chamfer_distance
    
    def __call__(self, pred, target):
        return self.loss_fn(pred, target)[0]

class earth_mover_distance:
    def __init__(self, train=True, feature_loss=torch.nn.MSELoss()):
        self.loss_fn = emdModule()
        self.eps = 0.005 if train else 0.002
        self.iterations = 50 if train else 10000
        self.feature_loss = feature_loss
    
    def __call__(self, pred, target):
        point_l = torch.sqrt(self.loss_fn(pred[:, :, :3], target[:, :, :3],  self.eps, self.iterations)[0]).mean()
        feature_l = self.feature_loss(pred[:, :, 3:], target[:, :, 3:])
        return point_l + feature_l
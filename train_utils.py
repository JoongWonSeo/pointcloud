import torch
import torch.nn
from pytorch3d.ops import sample_farthest_points
from pytorch3d import loss as pytorch3d_loss
from loss.emd.emd_module import emdModule
import numpy as np



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
    def __init__(self, bbox=None):
        self.loss_fn = pytorch3d_loss.chamfer_distance
        self.bbox = bbox
    
    def __call__(self, pred, target):
        if self.bbox is None:
            return self.loss_fn(pred, target)[0]

class earth_mover_distance:
    def __init__(self, eps = 0.002, iterations = 10000, feature_loss=torch.nn.MSELoss(), bbox=None, bbox_bonus=2):
        self.loss_fn = emdModule()
        self.eps = eps
        self.iterations = iterations
        self.feature_loss = feature_loss
        self.bbox = bbox
        self.bbox_bonus = bbox_bonus
    
    def __call__(self, pred, target):
        dists, assignment = self.loss_fn(pred[:, :, :3], target[:, :, :3],  self.eps, self.iterations)

        # compare the features (RGB) OF THE CORRESPONDING POINTS ACCORDING TO assignment
        assignment = assignment.long().unsqueeze(-1)
        target = target.take_along_dim(assignment, 1)
        feature_l = self.feature_loss(pred[:, :, 3:], target[:, :, 3:])

        # DEBUG: check the number of unassigned points
        num_points = pred.shape[1]
        num_missing = num_points - assignment.unique().numel()
        if num_missing > 0:
            print(f"unassigned = {num_missing} / {num_points} = {num_missing / num_points}")


        # sanity check: compare the points of the permuted pc
        # d = (pred[:,:,:3] - target[:,:,:3]) * (pred[:,:,:3] - target[:,:,:3])
        # d = torch.sqrt(d.sum(-1)).mean()

        # check if target is in bbox and increase weight of loss
        weights = torch.ones_like(dists)
        if self.bbox is not None:
            is_in_bbox = (self.bbox[0] < target[:, :, 0]) & (target[:, :, 0] < self.bbox[1]) \
                       & (self.bbox[2] < target[:, :, 1]) & (target[:, :, 1] < self.bbox[3]) \
                       & (self.bbox[4] < target[:, :, 2]) & (target[:, :, 2] < self.bbox[5])

            # weight the loss of points that are not in the bbox
            weights += (self.bbox_bonus * is_in_bbox.float())

        point_l = (dists.sqrt() * weights).mean()


        return point_l + feature_l
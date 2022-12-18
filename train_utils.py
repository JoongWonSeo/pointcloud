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

        # else:
            

            # # compute the loss
            # loss_in = self.loss_fn(pred_in, target_in)[0]

            # return 2*loss_in + loss_out


class earth_mover_distance:
    def __init__(self, train=True, feature_loss=torch.nn.MSELoss(), bbox=None):
        self.loss_fn = emdModule()
        self.eps = 0.005 if train else 0.002
        self.iterations = 50 if train else 10000
        self.feature_loss = feature_loss
        self.bbox = bbox
    
    def __call__(self, pred, target):
        # if self.bbox is None:
        #     # weight per point-pair is just 1
        #     weight = torch.ones(pred.shape[0], pred.shape[1], target.shape[1]).to(pred.device)

        dists, assignment = self.loss_fn(pred[:, :, :3], target[:, :, :3],  self.eps, self.iterations)
        point_l = torch.sqrt(dists).mean()

        # compare the features (RGB) OF THE CORRESPONDING POINTS ACCORDING TO assignment

        # numpy version
        # assignment = assignment.cpu().numpy()
        # assignment = np.expand_dims(assignment, -1)
        # target = np.take_along_axis(target, assignment, axis = 1)

        # torch version
        assignment = assignment.long().unsqueeze(-1)
        target = torch.take_along_dim(target, assignment, 1)

        
        # sanity check: compare the points of the permuted pc
        # d = (pred[:,:,:3] - target[:,:,:3]) * (pred[:,:,:3] - target[:,:,:3])
        # d = torch.sqrt(d.sum(-1)).mean()
        # print(f'loss = {point_l}, d = {d}')

        feature_l = self.feature_loss(pred[:, :, 3:], target[:, :, 3:])
        return point_l + feature_l
import cfg
from functools import reduce
import torch
import torch.nn
from torch.nn import MSELoss
from pytorch3d.ops import sample_farthest_points
from pytorch3d import loss as pytorch3d_loss
from .loss.emd.emd_module import emdModule



########## Segmentation Visualization ##########

def seg_to_color(seg, classes):
    if type(classes[0][1]) is not torch.Tensor:
        classes = [(name, torch.Tensor(col)) for name, col in classes]

    color = torch.zeros(seg.shape[0], 3)
    
    N = len(classes)
    seg = seg.squeeze(1)
    seg = (seg*(N-1)).round().long()
    
    for i, (name, c) in enumerate(classes):
        color[seg == i, :] = c

    if cfg.debug: # DEBUG: show class distribution
        points_per_class = [(seg == i).sum() for i in range(N)]
        num_points = seg.shape[0]
        for i in range(N):
            print(f"DEBUG: class {classes[i][0]} = {points_per_class[i]} / {num_points} = {points_per_class[i] / num_points}")

    return color



########## Point Cloud Transformations for PyTorch Datasets ##########

class SampleRandomPoints:
    def __init__(self, K):
        self.K = K
    
    def __call__(self, points):
        # sample K points
        points = points[torch.randint(points.shape[0], (self.K,)), :]

        return points

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

class FilterBBox:
    def __init__(self, bbox):
        '''
        bbox: 3D bounding box of the point cloud [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        '''
        self.bbox = torch.Tensor(bbox)
    
    def __call__(self, points):
        # filter points outside of bounding box
        mask = (points[:, 0] > self.bbox[0, 0]) & (points[:, 0] < self.bbox[0, 1]) \
             & (points[:, 1] > self.bbox[1, 0]) & (points[:, 1] < self.bbox[1, 1]) \
             & (points[:, 2] > self.bbox[2, 0]) & (points[:, 2] < self.bbox[2, 1])
        return points[mask]

class FilterClasses:
    def __init__(self, whitelist, seg_dim, num_classes=len(cfg.classes)):
        '''
        whitelist: list of classes to remove
        num_classes: number of classes in the dataset
        seg_dim: dimension index of the segmentation label in the point cloud
        '''
        self.whitelist = whitelist
        self.num_classes = num_classes
        self.seg_dim = seg_dim
    
    def __call__(self, points):
        # only keep points that are in the whitelist
        seg = points[:, self.seg_dim] # (N,)
        seg = (seg*(self.num_classes-1)).round().long() # (N,)
        mask = reduce(torch.logical_or, [seg == v for v in self.whitelist])
        return points[mask, :]

class Normalize:
    def __init__(self, bbox, dim=3):
        '''
        bbox: 3D bounding box of the point cloud [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        '''
        bbox = torch.Tensor(bbox)
        self.min = bbox[:, 0]
        self.max = bbox[:, 1]
        self.dim = dim

    def __call__(self, points):
        # normalize points along each axis
        points[:, 0:self.dim] = (points[:, 0:self.dim] - self.min) / (self.max - self.min)
        return points



########## Loss Functions ##########

class ChamferDistance:
    def __init__(self, bbox=None):
        self.loss_fn = pytorch3d_loss.chamfer_distance
        self.bbox = bbox
    
    def __call__(self, pred, target):
        if self.bbox is None:
            return self.loss_fn(pred, target)[0]

class EarthMoverDistance:
    def __init__(self, eps = 0.002, iterations = 10000, feature_loss=MSELoss(), classes=None):
        self.loss_fn = emdModule()
        self.eps = eps
        self.iterations = iterations
        self.feature_loss = feature_loss
        self.classes = classes
    
    def __call__(self, pred, target):
        dists, assignment = self.loss_fn(pred[:, :, :3], target[:, :, :3],  self.eps, self.iterations)

        # compare the features (RGB) OF THE CORRESPONDING POINTS ACCORDING TO assignment
        assignment = assignment.long().unsqueeze(-1)
        target = target.take_along_dim(assignment, 1) # permute target according to assignment, such that matched points are at the same index
        feature_l = self.feature_loss(pred[:, :, 3:], target[:, :, 3:])

        # DEBUG: check the number of unassigned points
        if cfg.debug:
            num_points = pred.shape[1]
            num_missing = num_points - assignment.unique().numel()
            if num_missing > 0:
                print(f"DEBUG: EMD unassigned = {num_missing} / {num_points} = {num_missing / num_points}")

        # sanity check: compare the points of the permuted pc
        # d = (pred[:,:,:3] - target[:,:,:3]) * (pred[:,:,:3] - target[:,:,:3])
        # d = torch.sqrt(d.sum(-1)).mean()

        # check if target is in bbox and increase weight of loss
        # now that segmentation is available, use it to assign weights by class
        weights = torch.ones_like(dists)
        if self.classes is not None:
            N = len(self.classes)
            target_classes = (target[:, :, 3]*(N-1)).round()
            for idx, (_, w) in enumerate(self.classes):
                weights[target_classes == idx] = w
            
            # if cfg.debug:
            #     points_per_class = [(target_classes == i).sum()/25 for i in range(N)]
            #     num_points = 2048
            #     for i in range(N):
            #         print(f"DEBUG: EMD class {self.classes[i][0]} = {points_per_class[i]} / {num_points} = {points_per_class[i] / num_points}")

        
        point_l = (dists.sqrt() * weights) / weights.sum()

        return point_l + feature_l

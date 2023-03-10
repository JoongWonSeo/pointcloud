import cfg
from functools import reduce, wraps
import os
import numpy as np
import torch
import torch.nn
from torch.nn import MSELoss
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points
from pytorch3d import loss as pytorch3d_loss
from .loss.emd.emd_module import emdModule



########## Segmentation Visualization ##########

def get_class_points(points, seg, cls, N=len(cfg.classes)):
    '''
    points: (N, D) tensor of points
    seg: (N, 1) tensor of segmentation labels
    cls: class index of points to return
    N: number of total classes in the segmentation
    '''

    seg = (seg*(N-1)).round().long().squeeze(1) # (N)

    return points[seg == cls, :] # (M, D)

    

def seg_to_color(seg, classes=cfg.classes):
    if type(classes[0][1]) is not torch.Tensor:
        classes = [(name, torch.Tensor(col)) for name, col in classes]
    if type(seg) is not torch.Tensor:
        seg = torch.from_numpy(seg)

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

# but first, a simple decorator to automatically convert numpy arrays to tensors and back
def support_numpy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        has_numpy = any([type(arg) is np.ndarray for arg in args]) or any([type(v) is np.ndarray for v in kwargs.values()])
        args = [torch.from_numpy(arg) if type(arg) is np.ndarray else arg for arg in args]
        kwargs = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in kwargs.items()}

        result = func(*args, **kwargs)
        return result.numpy() if has_numpy else result
    return wrapper


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

    @support_numpy
    def __call__(self, points):
        orig_shape = points.shape
        points = points.reshape(-1, points.shape[-1])
        # normalize points along each axis
        points[:, 0:self.dim] = (points[:, 0:self.dim] - self.min) / (self.max - self.min)
        return points.reshape(orig_shape)

class Unnormalize:
    def __init__(self, bbox, dim=3):
        '''
        bbox: 3D bounding box of the point cloud [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        '''
        bbox = torch.Tensor(bbox)
        self.min = bbox[:, 0]
        self.max = bbox[:, 1]
        self.dim = dim

    @support_numpy
    def __call__(self, points):
        # unnormalize points along each axis
        points[:, 0:self.dim] = points[:, 0:self.dim] * (self.max - self.min) + self.min
        return points

@support_numpy
def mean_cube_pos(Y):
    cube_points = get_class_points(Y[:, :3], Y[:, 3:4], 1, len(cfg.classes))

    if cfg.debug:
        if cube_points.shape[0] == 0:
            print("DEBUG: no cube points found")
            return torch.zeros(3)

    return cube_points.mean(dim=0)




########## Loss Functions ##########

class ChamferDistance:
    def __init__(self, bbox=None):
        self.loss_fn = pytorch3d_loss.chamfer_distance
        self.bbox = bbox
    
    def __call__(self, pred, target):
        if self.bbox is None:
            return self.loss_fn(pred, target)[0]

class EarthMoverDistance:
    def __init__(self, eps = 0.002, its = 10000, feature_loss=MSELoss(), classes=None):
        self.loss_fn = emdModule()
        self.eps = eps
        self.iterations = its
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

        
        point_l = (dists.sqrt() * weights).sum() / weights.sum()

        return point_l + feature_l



########## PyTorch Datasets ##########

class PointCloudDataset(Dataset):
    '''
    Dataset for point cloud to point cloud pairs, e.g. for training a point cloud autoencoder
    '''

    def __init__(self, root_dir, files=None, in_features=['rgb'], out_features=['rgb'], in_transform=None, out_transform=None):
        self.root_dir = root_dir

        # you can either pass a list of files or None for all files in the root_dir
        self.files = files if files is not None else os.listdir(root_dir)
        self.files = [f for f in self.files if f.endswith('.npz')] # get only npz files
        
        self.in_transform = in_transform
        self.out_transform = out_transform

        self.in_features = in_features
        self.out_features = out_features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pointcloud = self.get_file(idx)

        if self.in_features == self.out_features:
            features = [pointcloud[f] for f in self.in_features]
            in_pc = out_pc = torch.from_numpy(np.concatenate((pointcloud['points'], *features), axis=1))
        else:
            in_features = [pointcloud[f] for f in self.in_features]
            out_features = [pointcloud[f] for f in self.out_features]

            in_pc = torch.from_numpy(np.concatenate((pointcloud['points'], *in_features), axis=1))
            out_pc = torch.from_numpy(np.concatenate((pointcloud['points'], *out_features), axis=1))

        if self.in_transform:
            in_pc = self.in_transform(in_pc)
        if self.out_transform:
            out_pc = self.out_transform(out_pc)

        return in_pc, out_pc
    
    def filename(self, idx):
        return self.files[idx]

    def get_file(self, idx):
        return np.load(os.path.join(self.root_dir, self.files[idx]), allow_pickle=True)
    

class PointCloudGTDataset(Dataset):
    '''
    Dataset for point cloud to ground truth pairs, e.g. for training a point cloud to ground truth predictor
    '''

    def __init__(self, root_dir, files=None, in_features=['rgb'], in_transform=None, out_transform=None):
        self.root_dir = root_dir

        # you can either pass a list of files or None for all files in the root_dir
        self.files = files if files is not None else os.listdir(root_dir)
        self.files = [f for f in self.files if f.endswith('.npz')] # get only npz files
        
        self.in_transform = in_transform
        self.out_transform = out_transform

        self.in_features = in_features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pointcloud = self.get_file(idx)

        in_features = [pointcloud[f] for f in self.in_features]
        out_data = torch.from_numpy(pointcloud['ground_truth']).float()

        in_pc = torch.from_numpy(np.concatenate((pointcloud['points'], *in_features), axis=1))

        if self.in_transform:
            in_pc = self.in_transform(in_pc)
        if self.out_transform:
            out_data = self.out_transform(out_data)

        return in_pc, out_data
    
    def filename(self, idx):
        return self.files[idx]

    def get_file(self, idx):
        return np.load(os.path.join(self.root_dir, self.files[idx]), allow_pickle=True)
import pointcloud_vision.cfg as cfg
from functools import reduce, wraps
import os
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points
from pytorch3d import loss as pytorch3d_loss
from .loss.emd.emd_module import emdModule



########## Segmentation Visualization ##########

def get_class_points(points, seg, cls):
    '''
    points: (N, D) tensor of points
    seg: (N, 1) LongTensor of segmentation labels (integer encoded)
    cls: class index of points to return
    N: number of total classes in the segmentation
    '''

    seg = seg.long().squeeze(1) # (N)
    return points[seg == cls, :] # (M, D)

    

def seg_to_color(seg, classes):
    if type(classes[0][1]) is not torch.Tensor:
        classes = [(name, torch.Tensor(col)) for name, col in classes]
    if type(seg) is not torch.Tensor:
        seg = torch.from_numpy(seg)

    color = torch.zeros(seg.shape[0], 3)
    
    N = len(classes)
    seg = seg.squeeze(1).long()
    
    for i, (name, c) in enumerate(classes):
        color[seg == i, :] = c

    # if cfg.debug: # DEBUG: show class distribution
    #     points_per_class = [(seg == i).sum() for i in range(N)]
    #     num_points = seg.shape[0]
    #     for i in range(N):
    #         print(f"DEBUG: class {classes[i][0]} = {points_per_class[i]} / {num_points} = {points_per_class[i] / num_points}")

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
    
    @support_numpy
    def __call__(self, points):
        # sample K points
        points = points[torch.randint(points.shape[0], (self.K,)), :]

        return points.float()

class SampleFurthestPoints:
    def __init__(self, K):
        self.K = K
    
    @support_numpy
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
        self.bbox = torch.as_tensor(bbox)
    
    def __call__(self, points):
        # filter points outside of bounding box
        mask = (points[:, 0] > self.bbox[0, 0]) & (points[:, 0] < self.bbox[0, 1]) \
             & (points[:, 1] > self.bbox[1, 0]) & (points[:, 1] < self.bbox[1, 1]) \
             & (points[:, 2] > self.bbox[2, 0]) & (points[:, 2] < self.bbox[2, 1])
        return points[mask]

class FilterClasses:
    def __init__(self, whitelist, label_dim):
        '''
        whitelist: list of classes (labels) to keep
        num_classes: number of classes in the dataset
        label_dim: dimension index of the segmentation label in the point cloud
        '''
        self.whitelist = whitelist
        self.label_dim = label_dim
    
    def __call__(self, points):
        # only keep points that are in the whitelist
        label = points[:, self.label_dim].long() # (N,)
        mask = reduce(torch.logical_or, [label == v for v in self.whitelist])
        return points[mask, :]

class Normalize:
    def __init__(self, bbox, dim=3):
        '''
        bbox: 3D bounding box of the point cloud [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        '''
        bbox = torch.as_tensor(bbox)
        self.min = bbox[:, 0]
        self.max = bbox[:, 1]
        self.dim = dim

    @support_numpy
    def __call__(self, points):
        self.min, self.max = self.min.to(points.device), self.max.to(points.device)
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
        bbox = torch.as_tensor(bbox)
        self.min = bbox[:, 0]
        self.max = bbox[:, 1]
        self.dim = dim

    @support_numpy
    def __call__(self, points):
        self.min, self.max = self.min.to(points.device), self.max.to(points.device)
        # unnormalize points along each axis
        points[:, 0:self.dim] = points[:, 0:self.dim] * (self.max - self.min) + self.min
        return points
    
class OneHotEncode:
    '''
    Convert integer encoded segmentation labels to one-hot encoded labels
    '''
    def __init__(self, num_classes, seg_dim=3):
        self.C = num_classes
        self.d = seg_dim
    
    @support_numpy
    def __call__(self, points):
        # convert segmentation label to one-hot encoding
        labels = points[:, self.d].long() # (N,)
        labels = F.one_hot(labels, self.C).float() # (N, C)
        return torch.cat([points[:, :self.d], labels, points[:, self.d+1:]], dim=1)

class IntegerEncode:
    '''
    Convert one-hot encoded segmentation labels to integer encoded labels
    '''
    def __init__(self, num_classes, seg_dim=3):
        self.C = num_classes
        self.d = seg_dim
    
    @support_numpy
    def __call__(self, points):
        # convert one-hot encoded segmentation label to integer encoding
        labels = points[:, self.d:self.d+self.C].argmax(dim=1).float() # (N,)
        labels = labels.unsqueeze(1) # (N, 1)
        return torch.cat([points[:, :self.d], labels, points[:, self.d+self.C:]], dim=1)

@support_numpy
def mean_cube_pos(Y):
    cube_points = get_class_points(Y[:, :3], Y[:, 3:4], 1)

    if cfg.debug:
        if cube_points.shape[0] == 0:
            print("DEBUG: no cube points found")
            return torch.zeros(3)

    # return cube_points.median(dim=0).values
    return cube_points.mean(dim=0)




########## Loss Functions ##########

class ChamferDistance:
    def __call__(self, pred, target):
        return pytorch3d_loss.chamfer_distance(pred, target)[0]

class FilteringChamferDistance:
    def __init__(self, filter):
        self.filter = filter

    def __call__(self, pred, target):
        device, dtype = pred.device, torch.float32 # unfortunately, pytorch3d only supports float32 
        pred = pred.to(dtype=dtype)

        # filter each point cloud in the batch individually
        filtered = [self.filter(p)[:, :3] for p in target]
        num_points = [p.shape[0] for p in filtered]
        max_points = max(num_points)
        # pad and stack filtered point clouds
        target = torch.stack([F.pad(p, (0, 0, 0, max_points - p.shape[0])) for p in filtered]).to(dtype=dtype)

        return pytorch3d_loss.chamfer_distance(pred, target, y_lengths=torch.tensor(num_points, device=device))[0]

class SegmentingChamferDistance:
    def __init__(self, class_labels):
        '''
        This loss function is for MultiFilterAE, which outputs C many filtered point clouds, one for each class
        Therefore the predicted output is {class_name: point clouds}, and the target is a single segmented point cloud,
        which this loss function will automatically filter for each class, and then compute the chamfer distance
        class_labels: dictionary mapping class names to their label (index) in the one-hot encoded segmentation label
        '''
        self.classs_losses = {c: FilteringChamferDistance(FilterClasses([l], label_dim=3)) for c, l in class_labels.items()}

    def __call__(self, pred, target):
        loss_per_class = torch.stack([loss(pred[c], target) for c, loss in self.classs_losses.items()])
        # weight_per_class = torch.ones_like(loss_per_class) / len(loss_per_class)
        return loss_per_class.sum()

class EarthMoverDistance:
    def __init__(self, eps = 0.002, its = 10000, num_classes=None, feature_weight=0.1):
        self.loss_fn = emdModule()
        self.eps = eps
        self.iterations = its
        self.C = num_classes
        self.feature_weight = feature_weight
    
    def __call__(self, pred, target):
        dists, assignment = self.loss_fn(pred[:, :, :3], target[:, :, :3],  self.eps, self.iterations)

        # compare the features (RGB) OF THE CORRESPONDING POINTS ACCORDING TO assignment
        assignment = assignment.long().unsqueeze(-1)
        target = target.take_along_dim(assignment, 1) # permute target according to assignment, such that matched points are at the same index

        # DEBUG: check the number of unassigned points
        if cfg.debug:
            num_points = pred.shape[1]
            num_missing = num_points - assignment.unique().numel()
            if num_missing/num_points > 0.005:
                print(f"DEBUG: EMD unassigned = {num_missing} / {num_points} = {num_missing / num_points}")

        # use segmentation data to assign weights by class
        weights = torch.ones_like(dists) # (B, N)
        if self.C is not None: # segmentation
            # get segmentation labels
            target_classes = target[:, :, 3].long() # (B, N)

            # automatically estimate weights for each class by looking at the distribution in the given batch
            distribution = torch.bincount(target_classes.view(-1), minlength=self.C) # (C,)
            distribution = distribution / distribution.sum()

            # distribution of the predicted classes
            pred_classes = pred[:, :, 3:].argmax(dim=2) # (B, N)
            pred_distribution = torch.bincount(pred_classes.view(-1), minlength=self.C) # (C,)
            pred_distribution = pred_distribution / pred_distribution.sum()

            # KL divergence between the two distributions
            kl_div = F.kl_div(F.log_softmax(pred_distribution, dim=0), F.softmax(distribution, dim=0), reduction='batchmean')

            class_weights = (1 / (distribution + 1e-4)) ** (1-0)  # inverse of distribution TODO try 1-distribution
            class_weights = class_weights / class_weights.sum()
            # if cfg.debug:
            #     print(f"DEBUG: EMD batch distribution = {distribution}")
            #     print(f"DEBUG: EMD batch class weights = {class_weights}")

            # assign weights according to class
            weights = class_weights[target_classes]
            
            # pred needs to be permuted from (B, N, C) to (B, C, N) for cross_entropy
            ce_l = F.cross_entropy(pred.permute(0, 2, 1)[:, 3:, :], target_classes, weight=class_weights)
            feature_l = 0.1 * ce_l #+ 100 * kl_div #TODO adjust
            self.log('train_loss/cross_entropy', ce_l)
            self.log('train_loss/kl_divergence', kl_div)

        else: # general feature loss
            feature_l = F.mse_loss(pred[:, :, 3:], target[:, :, 3:])

        
        point_l = (dists.sqrt() * weights).sum() / weights.sum()
        # point_l = (dists * weights).sum() / weights.sum() # weighted mean squared distance (squared EMD = wasserstein distance)
        self.log('train_loss/EMD', point_l)
        self.log('train_loss/feature', feature_l)

        return point_l + feature_l

class StatePredictionLoss:
    def __init__(self, states, transforms):
        self.state_losses = {s: F.mse_loss for s in states}
        self.t = transforms
        # if no transform is given, use identity
        for s in states:
            if s not in self.t:
                self.t[s] = lambda x: x
    
    def __call__(self, pred, target):
        return torch.stack([loss(pred[s], self.t[s](target[s])) for s, loss in self.state_losses.items()]).mean()


########## PyTorch Datasets ##########

def obs_to_pc(obs, features):
    pc = torch.cat((torch.as_tensor(obs['points']), *(torch.as_tensor(obs[f]) for f in features)), dim=1)
    return pc

class PointCloudDataset(Dataset):
    '''
    Dataset for point cloud to point cloud pairs, e.g. for training a point cloud autoencoder
    '''

    def __init__(self, root_dir, files=None, in_features=['rgb'], out_features=['rgb'], in_transform=None, out_transform=None):
        '''
        root_dir: directory containing the point cloud files
        files: list of files to use, if None, all files in root_dir are used
        in_features: list of features to be added to the input point cloud
        out_features: list of features to be added to the output point cloud
        in_transform: transform to apply to the input point cloud
        out_transform: transform to apply to the output point cloud

        Note: features in ['rgb', 'segmentation']
        '''

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
        obs = self.get_file(idx)

        if self.in_features == self.out_features:
            in_pc = out_pc = obs_to_pc(obs, self.in_features)

            if self.in_transform:
                in_pc = self.in_transform(in_pc)
            elif self.out_transform: # don't apply twice, since it's the same pc
                out_pc = self.out_transform(out_pc)
        else:
            in_pc = obs_to_pc(obs, self.in_features)
            out_pc = obs_to_pc(obs, self.out_features)

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

    def __init__(self, root_dir, files=None, in_features=['rgb'], in_transform=None, out_transform=None, swap_xy=False):
        self.root_dir = root_dir

        # you can either pass a list of files or None for all files in the root_dir
        self.files = files if files is not None else os.listdir(root_dir)
        self.files = [f for f in self.files if f.endswith('.npz')] # get only npz files
        
        self.in_transform = in_transform
        self.out_transform = out_transform

        self.in_features = in_features

        self.swap_xy = swap_xy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obs = self.get_file(idx)

        out_data = {s: torch.from_numpy(v).float() for (s, v) in obs['ground_truth']}
        in_pc = obs_to_pc(obs, self.in_features)

        if self.in_transform:
            in_pc = self.in_transform(in_pc)
        if self.out_transform:
            out_data = self.out_transform(out_data)

        return (in_pc, out_data) if not self.swap_xy else (out_data, in_pc)
    
    def filename(self, idx):
        return self.files[idx]

    def get_file(self, idx):
        return np.load(os.path.join(self.root_dir, self.files[idx]), allow_pickle=True)


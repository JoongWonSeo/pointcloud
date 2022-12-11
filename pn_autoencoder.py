import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from models.pointnet import PointNetEncoder
from openpoints.models.backbone.pointnext import PointNextEncoder
import numpy as np
import pandas as pd
from openpoints.utils import EasyConfig


class PNAutoencoder(nn.Module):
    def __init__(self, out_points = 2048, dim_per_point=3):
        super().__init__()

        self.out_points = out_points
        self.dim_per_point = dim_per_point

        #self.encoder = PointNetEncoder(in_channels=dim_per_point)
        cfg = EasyConfig()
        cfg.load('pointnext-b.yaml', recursive=True)
        cfg.model.encoder_args.in_channels = dim_per_point
        cfg.model.in_channels = dim_per_point
        self.encoder = PointNextEncoder(**cfg.model.encoder_args)
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_points * dim_per_point),
        )
    
    def forward(self, X):
        embedding = self.encoder(X)
        return torch.reshape(self.decoder(embedding), (-1, self.out_points, self.dim_per_point))


class PointcloudDataset(Dataset):
    def __init__(self, root_dir, files=None, transform=None):
        self.root_dir = root_dir

        # you can either pass a list of files or None for all files in the root_dir
        self.files = files if files is not None else os.listdir(root_dir)
        self.files = [f for f in self.files if f.endswith('.npz')] # get only npz files
        
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pointcloud = np.load(os.path.join(self.root_dir, self.files[idx]))
        pointcloud = np.hstack((pointcloud['points'], np.mean(pointcloud['rgb'], axis=1).reshape((-1, 1)))).astype(np.float32)

        if self.transform:
            pointcloud = self.transform(pointcloud)

        return pointcloud
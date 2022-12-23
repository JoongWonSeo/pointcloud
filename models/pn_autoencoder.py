import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.pointnet import PointNetEncoder
import numpy as np

class PNAutoencoder(nn.Module):
    def __init__(self, out_points = 2048, dim_per_point=3):
        super().__init__()

        self.out_points = out_points
        self.dim_per_point = dim_per_point
        pe = PointNetEncoder(in_channels=dim_per_point, track_stats=True)
        # self.encoder = nn.Sequential(
        #     pe,
        #     nn.ReLU(),
        #     nn.Linear(pe.out_channels, 3),
        #     # nn.Tanh(), # normalize the embedding to [-1, 1] # DO NOT DO TANH
        # )
        self.encoder = pe
        self.decoder = nn.Sequential(
            # nn.Linear(3, 1024),
            nn.Linear(pe.out_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_points * dim_per_point),
            nn.Sigmoid(),
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
        pointcloud = torch.tensor(np.hstack((pointcloud['points'], pointcloud['features'])).astype(np.float32))

        if self.transform:
            pointcloud = self.transform(pointcloud)

        return pointcloud
    
    def filename(self, idx):
        return self.files[idx]
    
    def save(self, idx, path):
        pointcloud = self.__getitem__(idx)
        np.savez(os.path.join(path, self.files[idx]), points=pointcloud[:, :3], features=pointcloud[:, 3:])
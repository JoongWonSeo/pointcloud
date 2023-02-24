import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.pointnet import PointNetEncoder
import numpy as np

class PNAutoencoder(nn.Module):
    def __init__(self, out_points = 2048, in_dim=6, out_dim=6):
        super().__init__()

        self.out_points = out_points
        self.in_dim = in_dim
        self.out_dim = out_dim
        pe = PointNetEncoder(in_channels=self.in_dim, track_stats=True)
        # self.encoder = nn.Sequential(
        #     pe,
        #     nn.ReLU(),
        #     nn.Linear(pe.out_channels, 3),
        #     # nn.Tanh(), # normalize the embedding to [-1, 1] # DO NOT DO TANH
        # )
        self.encoder = pe
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_points * self.out_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, X):
        embedding = self.encoder(X)
        return self.decoder(embedding).reshape((-1, self.out_points, self.out_dim))


class PointcloudDataset(Dataset):
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
        pointcloud = np.load(os.path.join(self.root_dir, self.files[idx]))

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
    
    # def save(self, idx, path):
    #     pointcloud = self[idx]
    #     np.savez(os.path.join(path, self.files[idx]), points=pointcloud[:, :3], features=pointcloud[:, 3:])

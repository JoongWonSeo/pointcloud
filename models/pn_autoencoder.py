import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet import PointNetEncoder

class PNAutoencoder(nn.Module):
    def __init__(self, out_points = 2048, dimensions_per_point=3):
        super().__init__()

        self.out_points = out_points
        self.dim_per_point = dimensions_per_point

        self.encoder = PointNetEncoder(in_channels=dimensions_per_point)
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_points * dimensions_per_point),
        )
    
    def forward(self, X):
        embedding = self.encoder(X)
        return torch.reshape(self.decoder(embedding), (-1, self.out_points, self.dim_per_point))

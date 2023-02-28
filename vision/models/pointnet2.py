import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction


class PointNet2Encoder(nn.Module):
    def __init__(self, space_dims=3, feature_dims=3):
        super(PointNet2Encoder, self).__init__()
        
        self.space_dims = space_dims
        self.feature_dims = feature_dims

        in_channel = space_dims + feature_dims

        # PointNet++ Encoder
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)


    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1) # (B, N, C) -> (B, C, N)

        B, _, _ = xyz.shape
        if self.feature_dims > 0:
            norm = xyz[:, self.space_dims:, :]
            xyz = xyz[:, :self.space_dims, :]
        else:
            norm = None

        # PointNet++ Encoder
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)        

        return x


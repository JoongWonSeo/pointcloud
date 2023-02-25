import sys
import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points
from pytorch3d.vis.plotly_vis import plot_scene


if sys.argv[1].endswith('.npz'):
    pointcloud = np.load(sys.argv[1])
    points = torch.Tensor(pointcloud['points']) # (N, 3)
    rgb = torch.Tensor(pointcloud['rgb']) # (N, 3)
    seg = torch.Tensor(pointcloud['segmentation']) # (N, 1)

    # segmentation to color converter
    seg = torch.cat([torch.zeros_like(seg), seg, torch.zeros_like(seg)], dim=1)

    pointcloud = Pointclouds(points=[points], features=[seg])

    # print(points.size())
    plot_scene({
        "Pointcloud": {
            sys.argv[1]: pointcloud,
        }
    }, pointcloud_marker_size=2).show()

else: #TODO update to new format like above
    # Load point cloud
    pointcloud1 = np.load(f'input/{sys.argv[1]}.npz')
    points1 = torch.Tensor(pointcloud1['points'])
    feat1 = torch.Tensor(pointcloud1['segmentation'])
    feat1 = torch.cat([torch.zeros_like(feat1), feat1, torch.zeros_like(feat1)], dim=1)

    pointcloud2 = np.load(f'output/{sys.argv[1]}.npz')
    points2 = torch.Tensor(pointcloud2['points'])
    feat2 = torch.Tensor(pointcloud2['segmentation'])
    feat2 = torch.cat([torch.zeros_like(feat2), feat2, torch.zeros_like(feat2)], dim=1)


    # offset points
    points1[:, 1] -= 0.5
    points2[:, 1] += 0.5


    pointcloud = Pointclouds(points=[points1, points2], features=[feat1, feat2])

    # print(points.size())
    plot_scene({
        "Pointcloud": {
            "input": pointcloud[0],
            "reconstructed": pointcloud[1]
        }
    }, pointcloud_marker_size=2).show()
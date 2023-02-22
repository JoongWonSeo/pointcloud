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
    seg_range = torch.max(seg) - torch.min(seg)
    seg = (seg - torch.min(seg)) / seg_range
    seg = torch.cat([seg, seg, seg], dim=1)
    print(seg)

    pointcloud = Pointclouds(points=[points], features=[seg])

    # print(points.size())
    plot_scene({
        "Pointcloud": {
            sys.argv[1]: pointcloud,
        }
    }, pointcloud_marker_size=2).show()

else: #TODO update to new format like above
    # Load point cloud
    pointcloud1 = np.load(f'prep/{sys.argv[1]}.npz')
    points1 = torch.Tensor(pointcloud1['points']) # (N, 3)
    rgb1 = torch.Tensor(pointcloud1['features']) # (N, 3)

    pointcloud2 = np.load(f'output/{sys.argv[1]}.npz')
    points2 = torch.Tensor(pointcloud2['points']) # (N, 3)
    rgb2 = torch.Tensor(pointcloud2['features']) # (N, 3)

    # offset points
    points1[:, 1] -= 0.5
    points2[:, 1] += 0.5


    pointcloud = Pointclouds(points=[points1, points2], features=[rgb1, rgb2])

    # print(points.size())
    plot_scene({
        "Pointcloud": {
            "input": pointcloud[0],
            "reconstructed": pointcloud[1]
        }
    }, pointcloud_marker_size=2).show()
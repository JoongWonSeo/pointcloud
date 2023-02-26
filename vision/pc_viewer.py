import sys
import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

classes = [
    ('env', [0, 0, 0]),
    ('cube', [1, 0, 0]),
    ('arm', [0.5, 0.5, 0.5]),
    ('base', [0, 1, 0]),
    ('gripper', [0, 0, 1]),
]
classes = [(n, torch.Tensor(c)) for n, c in classes]

def seg_to_color(seg, classes):
    color = torch.zeros(seg.shape[0], 3)
    
    N = len(classes)
    seg = seg.squeeze(1)
    seg = (seg*(N-1)).round().long()
    
    for i, (name, c) in enumerate(classes):
        color[seg == i, :] = c

    
    points_per_class = [(seg == i).sum() for i in range(N)]
    num_points = seg.shape[0]
    for i in range(N):
        print(f"class {classes[i][0]} = {points_per_class[i]} / {num_points} = {points_per_class[i] / num_points}")
    return color

def split_by_class(points, seg, classes):
    N = len(classes)
    seg = seg.squeeze(1)
    seg = (seg*(N-1)).round().long()

    split_points = [points[seg == i, :] for i in range(N)]
    split_colors = [torch.tensor(classes[i][1]).repeat(split_points[i].shape[0], 1) for i in range(N)]
    pcs = Pointclouds(points=split_points, features=split_colors)

    return {name: pcs[i] for i, (name, _) in enumerate(classes)}

if sys.argv[1].endswith('.npz'):
    pointcloud = np.load(sys.argv[1])
    points = torch.Tensor(pointcloud['points']) # (N, 3)
    # rgb = torch.Tensor(pointcloud['rgb']) # (N, 3)
    seg = torch.Tensor(pointcloud['segmentation'])

    pointclouds = split_by_class(points, seg, classes)

    plot_scene({
        sys.argv[1]: pointclouds
    }, pointcloud_marker_size=2).show()

else:
    # Load point cloud
    pointcloud1 = np.load(f'input/{sys.argv[1]}.npz')
    points1 = torch.Tensor(pointcloud1['points'])
    feat1 = torch.Tensor(pointcloud1['segmentation'])

    pointcloud2 = np.load(f'output/{sys.argv[1]}.npz')
    points2 = torch.Tensor(pointcloud2['points'])
    feat2 = torch.Tensor(pointcloud2['segmentation'])

    # offset points
    points1[:, 1] -= 0.5
    points2[:, 1] += 0.5

    pcs1 = split_by_class(points1, feat1, classes)
    pcs1 = {'input_'+name: pc for name, pc in pcs1.items()}
    pcs2 = split_by_class(points2, feat2, classes)
    pcs2 = {'output_'+name: pc for name, pc in pcs2.items()}

    plot_scene({
        sys.argv[1]: pcs1 | pcs2
    }, pointcloud_marker_size=2).show()
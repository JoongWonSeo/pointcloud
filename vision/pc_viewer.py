import cfg
import sys
import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene


def seg_to_color(seg, classes):
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

def split_by_class(points, seg, classes):
    N = len(classes)
    seg = seg.squeeze(1)
    seg = (seg*(N-1)).round().long()

    split_points = [points[seg == i, :] for i in range(N)]
    split_colors = [classes[i][1].repeat(split_points[i].shape[0], 1) for i in range(N)]
    pcs = Pointclouds(points=split_points, features=split_colors)

    return {name: pcs[i] for i, (name, _) in enumerate(classes) if pcs[i].num_points_per_cloud()[0] > 0}

if sys.argv[1].endswith('.npz'):
    pointcloud = np.load(sys.argv[1], allow_pickle=True)
    points = torch.Tensor(pointcloud['points']) # (N, 3)
    seg = torch.Tensor(pointcloud['segmentation'])
    # rgb = torch.Tensor(pointcloud['rgb']) # (N, 3)
    classes = [(name, torch.Tensor(col)) for name, col in pointcloud['classes']]

    pointclouds = split_by_class(points, seg, classes)

    plot_scene({
        sys.argv[1]: pointclouds
    }, pointcloud_marker_size=2).show()

else:

    # Load point cloud
    pointcloud1 = np.load(f'input/{sys.argv[1]}.npz', allow_pickle=True)
    points1 = torch.Tensor(pointcloud1['points'])
    rgb1 = torch.Tensor(pointcloud1['rgb'])
    feat1 = torch.Tensor(pointcloud1['segmentation'])
    classes1 = [(name, torch.Tensor(col)) for name, col in pointcloud1['classes']]


    pointcloud2 = np.load(f'output/{sys.argv[1]}.npz', allow_pickle=True)
    points2 = torch.Tensor(pointcloud2['points'])
    feat2 = torch.Tensor(pointcloud2['segmentation'])
    classes2 = [(name, torch.Tensor(col)) for name, col in pointcloud2['classes']]

    # offset points
    points1[:, 1] -= 0.5
    points2[:, 1] += 0.5

    pcs1 = split_by_class(points1, feat1, classes1)
    pcs1 |= {'rgb': Pointclouds(points=[points1], features=[rgb1])}
    pcs1 = {'input_'+name: pc for name, pc in pcs1.items()}
    pcs2 = split_by_class(points2, feat2, classes2)
    pcs2 = {'output_'+name: pc for name, pc in pcs2.items()}

    plot_scene({
        sys.argv[1]: pcs1 | pcs2
    }, pointcloud_marker_size=2).show()
import argparse
import pointcloud_vision.cfg as cfg
import sys
import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene


def split_by_class(points, seg, classes):
    if type(classes[0][1]) is not torch.Tensor:
        classes = [(name, torch.Tensor(col)) for name, col in classes]

    N = len(classes)
    seg = seg.squeeze(1).long()

    split_points = [points[seg == i, :] for i in range(N)]
    split_colors = [classes[i][1].repeat(split_points[i].shape[0], 1) for i in range(N)]
    pcs = Pointclouds(points=split_points, features=split_colors)

    if cfg.debug:
        # print number of cube points
        num_cube = split_points[1].shape[0]
        print(f"DEBUG: cube points = {num_cube} / {points.shape[0]} = {num_cube / points.shape[0]}")

    return {name: pcs[i] for i, (name, _) in enumerate(classes) if pcs[i].num_points_per_cloud()[0] > 0}

def load_pointcloud(file_dir):
    return np.load(file_dir, allow_pickle=True)

def plot_pointcloud(pointcloud, name='pointcloud', max_points=20000, point_size=2):
    points = torch.Tensor(pointcloud['points']) # (N, 3)

    pointclouds = {}

    if 'segmentation' in pointcloud:
        classes = [(name, torch.Tensor(col)) for name, col in pointcloud['classes']]
        seg = torch.Tensor(pointcloud['segmentation'])
        pointclouds |= split_by_class(points, seg, classes)
    if 'rgb' in pointcloud:
        pointclouds |= {'rgb': Pointclouds(points=[points], features=[torch.Tensor(pointcloud['rgb'])])}

    plot_scene({
        name: pointclouds
    },
        pointcloud_marker_size=point_size,
        pointcloud_max_points=max_points,
    ).show()


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Pointcloud Viewer')
    parser.add_argument('file_dir', type=str, help='pointcloud file directory')
    parser.add_argument('--point_size', default=2, type=int, help='point size')
    parser.add_argument('--max_points', default=20000, type=int, help='max points')
    args = parser.parse_args()

    # plot
    plot_pointcloud(load_pointcloud(args.file_dir), name=args.file_dir, max_points=args.max_points, point_size=args.point_size)
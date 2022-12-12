import sys
import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

# Load point cloud
pointcloud = np.load(sys.argv[1])
points = torch.Tensor(pointcloud['points'])
rgb = torch.Tensor(pointcloud['rgb'])

pointcloud = Pointclouds(points=[points], features=[rgb])

print(points.size())
plot_scene({
    "Pointcloud": {
        "cloud": pointcloud
    }
}, pointcloud_marker_size=2).show()
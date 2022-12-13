import sys
import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points
from pytorch3d.vis.plotly_vis import plot_scene

# Load point cloud
pointcloud = np.load(sys.argv[1])
points = torch.Tensor(pointcloud['points']) # (N, 3)
rgb = torch.Tensor(pointcloud['features']) # (N, 3)

# merge points and rgb
# points = torch.cat((points, rgb), dim=1) # (N, 6)
# # convert to batch of 1 pointcloud
# points = points.unsqueeze(0) # (1, N, 6)
# # sample 2048 points
# points, _ = sample_farthest_points(points, K=2048)
# # convert back to (N, 6)
# points = points.squeeze(0) # (N, 6)
# # split points and rgb
# points, rgb = points.split([3, 3], dim=1) # (N, 3), (N, 3)

pointcloud = Pointclouds(points=[points], features=[rgb])

print(points.size())
plot_scene({
    "Pointcloud": {
        "cloud": pointcloud
    }
}, pointcloud_marker_size=2).show()
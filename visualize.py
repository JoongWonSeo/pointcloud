import sys
import torch
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f'device = {device}')


# Load point cloud
df = pd.read_csv(sys.argv[1])
verts = torch.Tensor(df.loc[:, ['y', 'x', 'z']].to_numpy()).to(device)

# point_cloud = Pointclouds(points=[verts], features=[rgb])
point_cloud = Pointclouds(points=[verts])

plot_scene({
    "Pointcloud": {
        "person": point_cloud
    }
}).show()
import open3d as o3d
import numpy as np
import time
import torch
from models.pn_autoencoder import PNAutoencoder
    

# load model
ae = PNAutoencoder(2048, 6)
ae.load_state_dict(torch.load('weights/last.pth'))
ae = ae.to('cuda')
ae.eval()


# define functions to update pointcloud and change x, y, z
x, y, z = 0, 0, 0

def make_xyz_changer(axis, inc):
    def xyz_changer(vis):
        global x, y, z
        if axis == 'x':
            x += inc
        elif axis == 'y':
            y += inc
        elif axis == 'z':
            z += inc
        else:
            raise ValueError(f'invalid axis: {axis}')
        print(f'{axis} = {x if axis == "x" else y if axis == "y" else z}')
        update_pointcloud()
        return True
    return xyz_changer

def update_pointcloud():
    global x, y, z
    embedding = torch.Tensor([x, y, z]).to('cuda')
    decoded =  torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point)).detach().cpu().numpy()
    points = decoded[0, :, :3]
    rgb = decoded[0, :, 3:]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return True

# create visualizer and window.
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(height=480, width=640)

# a, s, d to increment x, y, z; z, x, c to decrement x, y, z; using GLFW key codes
vis.register_key_callback(65, make_xyz_changer('x', 0.1))
vis.register_key_callback(83, make_xyz_changer('y', 0.1))
vis.register_key_callback(68, make_xyz_changer('z', 0.1))
vis.register_key_callback(90, make_xyz_changer('x', -0.1))
vis.register_key_callback(88, make_xyz_changer('y', -0.1))
vis.register_key_callback(67, make_xyz_changer('z', -0.1))

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()
update_pointcloud() # initial points
vis.add_geometry(pcd) # include it in the visualizer before non-blocking visualization.

# run non-blocking visualization. 
keep_running = True
while keep_running:
    keep_running = vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
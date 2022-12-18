import open3d as o3d
import numpy as np
import time
import torch
from models.pn_autoencoder import PNAutoencoder, PointcloudDataset

x, y, z = 0, 0, 0

def increment_x(vis):
    global x
    x += 0.1
    print(f'x = {x}')
    update_pointcloud(vis)
    return True

def increment_y(vis):
    global y
    y += 0.1
    print(f'y = {y}')
    update_pointcloud(vis)
    return True

def increment_z(vis):
    global z
    z += 0.1
    print(f'z = {z}')
    update_pointcloud(vis)
    return True

def decrement_x(vis):
    global x
    x -= 0.1
    print(f'x = {x}')
    update_pointcloud(vis)
    return True

def decrement_y(vis):
    global y
    y -= 0.1
    print(f'y = {y}')
    update_pointcloud(vis)
    return True

def decrement_z(vis):
    global z
    z -= 0.1
    print(f'z = {z}')
    update_pointcloud(vis)
    return True


def update_pointcloud(vis):
    global x, y, z
    embedding = torch.Tensor([x, y, z]).to('cuda')
    points = torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point))[0, :, :3].detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    return True
    

# load model
ae = PNAutoencoder(2048, 6)
ae.load_state_dict(torch.load('weights/last.pth'))
ae = ae.to('cuda')
ae.eval()


# create visualizer and window.
# vis = o3d.visualization.Visualizer()
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(height=480, width=640)

# a, s, d to increment x, y, z, using GLFW key codes
vis.register_key_callback(65, increment_x)
vis.register_key_callback(83, increment_y)
vis.register_key_callback(68, increment_z)

# z, x, c to decrement x, y, z
vis.register_key_callback(90, decrement_x)
vis.register_key_callback(88, decrement_y)
vis.register_key_callback(67, decrement_z)

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()
# *optionally* add initial points
embedding = torch.Tensor([0.0, 0.0, 0.0]).to('cuda')
points = torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point))[0, :, :3].detach().cpu().numpy()
pcd.points = o3d.utility.Vector3dVector(points)

# load the preprocessed scene
# scene = o3d.geometry.PointCloud()
# scene.points = o3d.utility.Vector3dVector(np.load('prep/0.npz')['points'])

# include it in the visualizer before non-blocking visualization.
vis.add_geometry(pcd)
# vis.add_geometry(scene)

# to add new points each dt secs.
dt = 0.01
# number of points that will be added
n_new = 10

previous_t = time.time()

# run non-blocking visualization. 
# To exit, press 'q' or click the 'x' of the window.
keep_running = True
while keep_running:
    
    # if time.time() - previous_t > dt:
    #     # Options (uncomment each to try them out):
    #     # 1) extend with ndarrays.
    #     # pcd.points.extend(np.random.rand(n_new, 3))
        
    #     # 2) extend with Vector3dVector instances.
    #     # pcd.points.extend(
    #     #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))
        
    #     # 3) other iterables, e.g
    #     # pcd.points.extend(np.random.rand(n_new, 3).tolist())

    #     # points = np.random.rand(10, 3)
    #     # pcd.points = o3d.utility.Vector3dVector(points)
        
    #     #random embedding
    #     embedding = torch.Tensor(np.random.rand(3)).to('cuda')
    #     # embedding = torch.Tensor([0.0, 0.0, 0.0]).to('cuda')
    #     points = torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point))[0, :, :3].detach().cpu().numpy()
    #     pcd.points = o3d.utility.Vector3dVector(points)

    #     vis.update_geometry(pcd)
    #     previous_t = time.time()

    keep_running = vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
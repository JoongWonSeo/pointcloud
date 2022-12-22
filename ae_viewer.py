import open3d as o3d
import numpy as np
import torch
from models.pn_autoencoder import PNAutoencoder, PointcloudDataset
    
def main(device='cuda', model_dir='weights/PC_AE.pth', input_dir='prep'):
    # load model
    ae = PNAutoencoder(2048, 6)
    ae.load_state_dict(torch.load(model_dir))
    ae = ae.to(device)
    ae.eval()


    # define functions to update pointcloud and change x, y, z
    x, y, z = 0, 0, 0

    def make_xyz_changer(axis, inc):
        def xyz_changer(vis):
            nonlocal x, y, z
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
        nonlocal x, y, z
        embedding = torch.Tensor([x, y, z]).to(device)
        decoded =  torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.dim_per_point)).detach().cpu().numpy()
        points = decoded[0, :, :3]
        rgb = decoded[0, :, 3:]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        return True

    # input pointcloud -> encoder -> latent variable -> decoder -> pointcloud
    input_index = 0
    input_set = PointcloudDataset(root_dir=input_dir)

    def update_input():
        nonlocal input_index
        # load input pointcloud
        orig = input_set[input_index]
        orig_points = orig[:, :3]
        orig_rgb = orig[:, 3:]

        orig = orig.to(device).reshape((1, ae.out_points, ae.dim_per_point))
        embedding = ae.encoder(orig)
        print(embedding)
        pred = ae.decoder(embedding).reshape((-1, ae.out_points, ae.dim_per_point)).detach().cpu().numpy()

        # shift the orig_points and pred points so they don't overlap
        orig_points[:, 1] -= 0.6
        pred[0, :, 1] += 0.6
        
        # merge input and output pointclouds
        points = np.concatenate((orig_points, pred[0, :, :3]), axis=0)
        rgb = np.concatenate((orig_rgb, pred[0, :, 3:]), axis=0)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        return True

    def make_input_changer(inc):
        def input_changer(vis):
            nonlocal input_index
            input_index += 1 if inc else -1
            print(f'input = {input_index}')
            update_input()
            return True
        return input_changer    

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

    # left, right to change input pointcloud
    vis.register_key_callback(262, make_input_changer(True))
    vis.register_key_callback(263, make_input_changer(False))


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


main()
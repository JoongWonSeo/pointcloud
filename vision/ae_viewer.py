import cfg
import argparse
import open3d as o3d
import numpy as np
import torch
from models.pn_autoencoder import PointcloudDataset, PNAutoencoder, PN2Autoencoder, PN2PosExtractor
from vision.utils import seg_to_color, mean_cube_pos

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='weights/PC_AE.pth')
parser.add_argument('--input', default='input')
arg = parser.parse_args()


def aa_lines(pos, col, length=0.4, res=50):
    # create axis-aligned lines to show the predicted cube coordinate
    pos = np.concatenate((pos, col.reshape((1, 3))), axis=1)
    for axis in range(3):
        for dist in range(res):
            linepoints = np.zeros((res, 6)) # 10 * [X, Y, Z, R, G, B]
            offset = np.zeros(3)
            offset[axis] = -length/2 + dist/res * length
            linepoints[dist, :3] = pos[0, :3] + offset
            linepoints[dist, 3:] = col

            pos = np.concatenate((pos, linepoints), axis=0)
    return pos


def main(model_dir, input_dir):
    # load model
    # ae = PNAutoencoder(2048, in_dim=6, out_dim=4)
    ae = cfg.create_autoencoder()
    ae.load_state_dict(torch.load(model_dir))
    ae = ae.to(cfg.device)
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
        embedding = torch.Tensor([x, y, z]).to(cfg.device)
        decoded =  torch.reshape(ae.decoder(embedding), (-1, ae.out_points, ae.out_dim)).detach().cpu().numpy()
        points = decoded[0, :, :3]
        rgb = decoded[0, :, 3:]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        return True

    # input pointcloud -> encoder -> latent variable -> decoder -> pointcloud
    input_index = 0
    input_current = None
    input_set = PointcloudDataset(**cfg.get_dataset_args(input_dir))

    def update_input():
        nonlocal input_index, input_current
        # load input pointcloud
        orig, target = input_set[input_index]

        pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

        # create axis-aligned lines to show the predicted cube coordinate
        vis = aa_lines(orig.unsqueeze(0), np.array([0, 1, 0]), res=50)
        
        # merge input and output pointclouds
        points = np.concatenate((pred[:, :3], vis[:, :3]), axis=0)
        rgb = np.concatenate((seg_to_color(pred[:, 3:], cfg.classes), vis[:, 3:]), axis=0)
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
    vis.register_key_callback(262, make_input_changer(inc=True))
    vis.register_key_callback(263, make_input_changer(inc=False))


    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    # update_pointcloud() # initial points
    update_input() # initial points
    vis.add_geometry(pcd) # include it in the visualizer before non-blocking visualization.

    # run non-blocking visualization. 
    keep_running = True
    while keep_running:
        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


main(arg.model, arg.input)
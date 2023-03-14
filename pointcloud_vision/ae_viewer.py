import pointcloud_vision.cfg as cfg
import argparse
import open3d as o3d
import numpy as np
import torch
from pointcloud_vision.train import create_model
from pointcloud_vision.utils import seg_to_color, mean_cube_pos, IntegerEncode

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='weights/PC_AE.pth')
parser.add_argument('--input', default='input')
arg = parser.parse_args()


def aa_lines(pos, col, length=0.4, res=50):
    # create axis-aligned lines to show the predicted cube coordinate
    pos = np.concatenate((pos.reshape(1, 3), col.reshape((1, 3))), axis=1)
    for axis in range(3):
        for dist in range(res):
            linepoints = np.zeros((res, 6)) # 10 * [X, Y, Z, R, G, B]
            offset = np.zeros(3)
            offset[axis] = -length/2 + dist/res * length
            linepoints[dist, :3] = pos[0, :3] + offset
            linepoints[dist, 3:] = col

            pos = np.concatenate((pos, linepoints), axis=0)
    return pos

def interpolate_transition(prev, next, interp):
    return prev * (1 - interp) + next * interp

def main(model_dir, input_dir):
    # load model
    ae, open_dataset = create_model('Segmenter', 'PointNet2', load_dir=model_dir)
    ae = ae.model

    ae = ae.to(cfg.device)
    ae.eval()

    # input pointcloud -> encoder -> latent variable -> decoder -> pointcloud
    input_index = 0
    curr_pc, prev_pc = None, None
    input_set = open_dataset(input_dir)
    anim_t = 0

    to_label = IntegerEncode()

    # PosDecoder
    # def load_pc(index):
    #     # load input pointcloud
    #     orig, target = input_set[index]

    #     pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

    #     # create axis-aligned lines to show the original cube coordinate
    #     vis = aa_lines(orig.unsqueeze(0), np.array([0, 1, 0]), res=50)
        
    #     # merge input and output pointclouds
    #     points = np.concatenate((pred[:, :3], vis[:, :3]), axis=0)
    #     rgb = np.concatenate((seg_to_color(pred[:, 3:]), vis[:, 3:]), axis=0)

    #     return points, rgb

    # PN2Autoencoder    
    def load_pc(index):
        # load input pointcloud
        orig, target = input_set[index]

        pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

        pred, target = to_label(pred), to_label(target)

        # create axis-aligned lines to show the predicted cube coordinate
        vis = aa_lines(mean_cube_pos(target), np.array([1, 0, 0]), res=50)
        vis = np.concatenate((vis, aa_lines(mean_cube_pos(pred), np.array([0, 1, 0]), res=50)), axis=0)

        # shift so they are next to each other
        target[:, 1] -= 0.5
        pred[:, 1] += 0.5
        vis[:, 1] -= 0.5 # show at orig
        
        # merge input and output pointclouds
        points = np.concatenate((target[:, :3], pred[:, :3], vis[:, :3]), axis=0)
        rgb = np.concatenate((seg_to_color(target[:, 3:]), seg_to_color(pred[:, 3:]), vis[:, 3:]), axis=0)

        return points, rgb


    def update_input():
        nonlocal prev_pc, curr_pc, input_index, anim_t
        prev_pc = curr_pc
        points, rgb = load_pc(input_index)
        curr_pc = points, rgb
        anim_t = 0
        
        return False # update geom is done in the main loop

    def make_input_changer(inc):
        def input_changer(vis):
            nonlocal input_index
            input_index += 1 if inc else -1
            print(f'input = {input_index}')
            return update_input()
        return input_changer    

    # create visualizer and window.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(height=480, width=640)

    # a, s, d to increment x, y, z; z, x, c to decrement x, y, z; using GLFW key codes
    # vis.register_key_callback(65, make_xyz_changer('x', 0.1))
    # vis.register_key_callback(83, make_xyz_changer('y', 0.1))
    # vis.register_key_callback(68, make_xyz_changer('z', 0.1))
    # vis.register_key_callback(90, make_xyz_changer('x', -0.1))
    # vis.register_key_callback(88, make_xyz_changer('y', -0.1))
    # vis.register_key_callback(67, make_xyz_changer('z', -0.1))

    # left, right to change input pointcloud
    vis.register_key_callback(262, make_input_changer(inc=True))
    vis.register_key_callback(263, make_input_changer(inc=False))


    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    # update_pointcloud() # initial points
    update_input() # initial points
    pcd.points = o3d.utility.Vector3dVector(curr_pc[0])
    pcd.colors = o3d.utility.Vector3dVector(curr_pc[1])
    vis.add_geometry(pcd) # include it in the visualizer before non-blocking visualization.

    # run non-blocking visualization. 
    keep_running = True
    while keep_running:
        keep_running = vis.poll_events()
        if prev_pc is not None and curr_pc is not None and anim_t < 1:
            anim_t = min(anim_t + 0.1, 1)
            points = interpolate_transition(prev_pc[0], curr_pc[0], anim_t)
            rgb = interpolate_transition(prev_pc[1], curr_pc[1], anim_t)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            vis.update_geometry(pcd)
        vis.update_renderer()

    vis.destroy_window()


main(arg.model, arg.input)
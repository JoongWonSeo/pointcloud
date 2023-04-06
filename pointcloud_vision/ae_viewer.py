import os
import pointcloud_vision.cfg as cfg
import argparse
import open3d as o3d
import numpy as np
import torch
from pointcloud_vision.train import create_model
from pointcloud_vision.utils import seg_to_color, mean_cube_pos, IntegerEncode
from types import SimpleNamespace
from robosuite_envs.envs import cfg_vision


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

def main(dataset, env, model, backbone='PointNet2', model_ver=-1, view_mode='overlap', animation_speed=0.1):
    env_cfg = SimpleNamespace(**cfg_vision[env]) # dot notation rather than dict notation

    input_dir = f'input/{dataset}/val'
    if model_ver > -1:
        model_dir = f'output/{dataset}/{model}_{backbone}/version_{model_ver}/checkpoints/'
    else:
        model_dir = f'output/{dataset}/{model}_{backbone}/'
        model_dir += sorted(map(lambda n: (len(n), n), os.listdir(model_dir)))[-1][1] # lastest version, sorted first by length and then by name
        model_dir += '/checkpoints/'
    model_dir += sorted(os.listdir(model_dir))[-1] # lastest checkpoint
    print('using input dataset', input_dir)
    print('loading model', model_dir)

    # load model
    ae, open_dataset = create_model(model, backbone, env, load_dir=model_dir)
    ae = ae.model

    ae = ae.to(cfg.device)
    ae.eval()

    if model == 'Segmenter':
        classes = env_cfg.classes
        C = len(classes)
        to_label = IntegerEncode(num_classes=C)

    # states
    input_set = open_dataset(input_dir) # dataset of input pointclouds
    input_index = 0 # index of the shown pointcloud
    curr_pc, prev_pc = None, None # for interpolation transition
    anim_t = 0 # for interpolation animation

    # Autoencoder Input Output
    def load_pc(index):
        if model == 'Autoencoder':
            # show the input and predicted pointclouds
            orig, target = input_set[index]

            pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

            if view_mode == 'sidebyside':
                # shift so they are next to each other
                target[:, 1] -= 0.3
                pred[:, 1] += 0.3

            if view_mode == 'overlap':
                # apply red tint and green tint
                target[:, 3:] = interpolate_transition(target[:, 3:], np.array([0, 1, 0]), 0.3)
                pred[:, 3:] = interpolate_transition(pred[:, 3:], np.array([1, 0, 0]), 0.3)

            # merge input and output pointclouds
            points = np.concatenate((orig[:, :3], pred[:, :3]), axis=0)
            rgb = np.concatenate((orig[:, 3:], pred[:, 3:]), axis=0)

            return points, rgb

        if model == 'Segmenter':
            # show the input and predicted labeled pointclouds  
            orig, target = input_set[index]

            pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

            pred, target = to_label(pred), to_label(target) # one hot to integer label

            # create axis-aligned lines to show the predicted cube coordinate
            vis = aa_lines(mean_cube_pos(target), np.array([0, 1, 0]), res=50)
            vis = np.concatenate((vis, aa_lines(mean_cube_pos(pred), np.array([1, 0, 0]), res=50)), axis=0)

            # convert to color
            pred = np.concatenate((pred[:, :3], seg_to_color(pred[:, 3:], classes)), axis=1)
            target = np.concatenate((target[:, :3], seg_to_color(target[:, 3:], classes)), axis=1)

            if view_mode == 'sidebyside':
                # shift so they are next to each other
                target[:, 1] -= 0.5
                pred[:, 1] += 0.5
                vis[:, 1] -= 0.5 # show at orig
            
            if view_mode == 'overlap':
                # apply red tint and green tint
                target[:, 3:] = interpolate_transition(target[:, 3:], np.array([0, 1, 0]), 0.3)
                pred[:, 3:] = interpolate_transition(pred[:, 3:], np.array([1, 0, 0]), 0.3)
            
            # merge input and output pointclouds
            points = np.concatenate((target[:, :3], pred[:, :3], vis[:, :3]), axis=0)
            rgb = np.concatenate((target[:, 3:], pred[:, 3:], vis[:, 3:]), axis=0)

            return points, rgb


    
        if model == 'GTEncoder':
            # show the input pointcloud and the predicted cube coordinate
            orig, target = input_set[index]

            pred = ae(orig.to(cfg.device).unsqueeze(0)).detach().cpu().numpy()

            # create axis-aligned lines to show the original cube coordinate
            target_pos = aa_lines(target, np.array([0, 1, 0]), res=50)
            pred_pos = aa_lines(pred, np.array([1, 0, 0]), res=50)
            vis = np.concatenate((target_pos, pred_pos), axis=0)
            
            # merge input and visualizations
            points = np.concatenate((orig[:, :3], vis[:, :3]), axis=0)
            rgb = np.concatenate((orig[:, 3:], vis[:, 3:]), axis=0)

            nonlocal animation_speed
            animation_speed = 1 # no animation, makes no sense for this

            return points, rgb


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

    # Segmenting Autoencoder Input Output
    # to_label = IntegerEncode()
    # def load_pc(index):
    #     # load input pointcloud
    #     orig, target = input_set[index]

    #     pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

    #     pred, target = to_label(pred), to_label(target)

    #     # create axis-aligned lines to show the predicted cube coordinate
    #     vis = aa_lines(mean_cube_pos(target), np.array([1, 0, 0]), res=50)
    #     vis = np.concatenate((vis, aa_lines(mean_cube_pos(pred), np.array([0, 1, 0]), res=50)), axis=0)

    #     # shift so they are next to each other
    #     target[:, 1] -= 0.5
    #     pred[:, 1] += 0.5
    #     vis[:, 1] -= 0.5 # show at orig
        
    #     # merge input and output pointclouds
    #     points = np.concatenate((target[:, :3], pred[:, :3], vis[:, :3]), axis=0)
    #     rgb = np.concatenate((seg_to_color(target[:, 3:]), seg_to_color(pred[:, 3:]), vis[:, 3:]), axis=0)

    #     return points, rgb


    def update_input():
        nonlocal prev_pc, curr_pc, input_index, anim_t
        prev_pc = curr_pc
        points, rgb = load_pc(input_index)
        curr_pc = points, rgb
        anim_t = 0 # start transition animation
        
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

    # left, right to change input pointcloud
    vis.register_key_callback(262, make_input_changer(inc=True))
    vis.register_key_callback(263, make_input_changer(inc=False))

    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    update_input() # initial points
    pcd.points = o3d.utility.Vector3dVector(curr_pc[0])
    pcd.colors = o3d.utility.Vector3dVector(curr_pc[1])
    vis.add_geometry(pcd) # include it in the visualizer before non-blocking visualization.

    # run non-blocking visualization. 
    keep_running = True
    while keep_running:
        keep_running = vis.poll_events()
        if prev_pc is not None and curr_pc is not None and anim_t < 1:
            anim_t = min(anim_t + animation_speed, 1)
            points = interpolate_transition(prev_pc[0], curr_pc[0], anim_t)
            rgb = interpolate_transition(prev_pc[1], curr_pc[1], anim_t)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            vis.update_geometry(pcd)
        vis.update_renderer()

    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('env', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--backbone', default='PointNet2', type=str)
    parser.add_argument('--model_ver', default=-1, type=int)
    parser.add_argument('--view', default='overlap', choices=['overlap', 'sidebyside'])
    parser.add_argument('--animation_speed', default=0.1, type=float)
    arg = parser.parse_args()

    main(arg.dataset, arg.env, arg.model, arg.backbone, arg.model_ver, arg.view, arg.animation_speed)
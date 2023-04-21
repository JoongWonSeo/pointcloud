import os
import pointcloud_vision.cfg as cfg
import argparse
import open3d as o3d
import numpy as np
import torch
from pointcloud_vision.train import create_model
from pointcloud_vision.utils import seg_to_color, mean_cube_pos, IntegerEncode
from pointcloud_vision.pc_encoder import StatePredictor
from types import SimpleNamespace
from robosuite_envs.envs import cfg_scene


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

def main(scene_name, model, backbone='PointNet2', model_ver=-1, view_mode='overlap', animation_speed=0.1):
    scene = SimpleNamespace(**cfg_scene[scene_name]) # dot notation rather than dict notation

    input_dir = f'input/{scene_name}/val'
    if model_ver > -1:
        model_dir = f'output/{scene_name}/{model}_{backbone}/version_{model_ver}/checkpoints/'
    else:
        model_dir = f'output/{scene_name}/{model}_{backbone}/'
        model_dir += sorted(map(lambda n: (len(n), n), os.listdir(model_dir)))[-1][1] # lastest version, sorted first by length and then by name
        model_dir += '/checkpoints/'
    model_dir += sorted(os.listdir(model_dir))[-1] # lastest checkpoint
    print('using input dataset', input_dir)
    print('loading model', model_dir)

    # load model
    ae, open_dataset = create_model(model, backbone, scene_name, load_dir=model_dir)
    ae = ae.model

    ae = ae.to(cfg.device)
    ae.eval()

    if model in ['Segmenter', 'GTSegmenter', 'MultiSegmenter']:
        classes = list(zip(scene.classes, scene.class_colors))
        C = len(classes)
        to_label = IntegerEncode(num_classes=C)
    if model == 'StatePredictor':
        from_state = StatePredictor.from_state(scene)

    # states
    input_set = open_dataset(input_dir) # dataset of input pointclouds
    input_index = 0 # index of the shown pointcloud
    curr_pc, prev_pc = None, None # for interpolation transition
    anim_t = 0 # for interpolation animation

    # Autoencoder Input Output
    def load_pc(index):
        nonlocal animation_speed, view_mode

        # load the dataset
        orig, target = input_set[index]

        target_pc, target_feature = None, None
        pred_pc, pred_feature = None, None
        target_gts, pred_gts = None, None

        if model == 'Autoencoder':
            # show the input and predicted pointclouds
            pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

            target_pc = target
            target_feature = 'rgb'
            pred_pc = pred
            pred_feature = 'rgb'

        if model == 'Segmenter':
            pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            pred = to_label(pred) # one hot to integer label (target is already integer label)

            target_pc = target
            target_feature = 'seg'
            pred_pc = pred
            pred_feature = 'seg'
            target_gts = [mean_cube_pos(target)]
            pred_gts = [mean_cube_pos(pred)]
        
        if model == 'MultiSegmenter':
            pred = ae.reconstruct_labeled(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            print('global encoding:', ae.global_encoding)
            print('encodings:', ae.local_encodings)

            target_pc = target
            target_feature = 'seg'
            pred_pc = pred
            pred_feature = 'seg'
            target_gts = [mean_cube_pos(target)]
            pred_gts = [mean_cube_pos(pred)]
        
        if model == 'StatePredictor':
            pred = ae(orig.to(cfg.device).unsqueeze(0))
            # print(pred, target)
            for k in pred:
                if k not in from_state:
                    print('using identity function for', k)
                    from_state[k] = lambda x: x
            pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}
            target = {k: from_state[k](v.detach().cpu().numpy()) for k, v in target.items()}

            target_pc = orig
            target_feature = 'rgb'
            if scene_name == 'Cube':
                target_gts = [target['cube_pos'], target['robot0_eef_pos']]
                pred_gts = [pred['cube_pos'], pred['robot0_eef_pos']]
            if scene_name == 'PegInHole':
                target_gts = [target['hole_pos'], target['hole_pos'] - from_state['hole_pos'](target['peg_to_hole'])]
                pred_gts = [pred['hole_pos'], pred['hole_pos'] - pred['peg_to_hole']]


        # assemble the pointclouds        
        if target_pc is not None and target_feature == 'seg': # convert to color
            target_pc = np.concatenate((target_pc[:, :3], seg_to_color(target_pc[:, 3:], classes)), axis=1)
        if pred_pc is not None and pred_feature == 'seg': # convert to color
            pred_pc = np.concatenate((pred_pc[:, :3], seg_to_color(pred_pc[:, 3:], classes)), axis=1)

        # create visualization
        vis = np.array([]).reshape(0, 6)
        if target_gts is not None:
            for i, gt in enumerate(target_gts):
                vis = np.concatenate((vis, aa_lines(gt, np.array([0, 1, i*0.5]), res=50)), axis=0)
        if pred_gts is not None:
            for i, gt in enumerate(pred_gts):
                vis = np.concatenate((vis, aa_lines(gt, np.array([1, 0, i*0.5]), res=50)), axis=0)
        
        # apply view mode
        if target_pc is not None and pred_pc is not None:
            if view_mode == 'sidebyside':
                # shift so they are next to each other
                target_pc[:, 1] -= 0.3
                pred_pc[:, 1] += 0.3
                vis[:, 1] -= 0.3 # show at orig

            if view_mode == 'overlap':
                # apply red tint and green tint
                target_pc[:, 3:] = interpolate_transition(target_pc[:, 3:], np.array([0, 1, 0]), 0.3)
                pred_pc[:, 3:] = interpolate_transition(pred_pc[:, 3:], np.array([1, 0, 0]), 0.3)
            
        # merge input and visualizations
        pcs = filter(lambda x: x is not None, [target_pc, pred_pc, vis])
        pcs = np.concatenate(tuple(pcs), axis=0)
        return pcs[:, :3], pcs[:, 3:]


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
    parser.add_argument('scene', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--backbone', default='PointNet2', type=str)
    parser.add_argument('--model_ver', default=-1, type=int)
    parser.add_argument('--view', default='overlap', choices=['overlap', 'sidebyside'])
    parser.add_argument('--animation_speed', default=0.1, type=float)
    arg = parser.parse_args()

    main(arg.scene, arg.model, arg.backbone, arg.model_ver, arg.view, arg.animation_speed)
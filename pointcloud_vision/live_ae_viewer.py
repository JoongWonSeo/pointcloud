import os
import pointcloud_vision.cfg as cfg
import argparse
import open3d as o3d
import numpy as np
import torch
import gymnasium as gym
from pointcloud_vision.pc_sensor import PointCloudSensor
from pointcloud_vision.pc_encoder import StatePredictor
from pointcloud_vision.train import create_model
from pointcloud_vision.utils import obs_to_pc, seg_to_color, mean_cube_pos, IntegerEncode, Normalize, Unnormalize
from robosuite_envs.envs import cfg_scene
from sb3_contrib.tqc.policies import MultiInputPolicy


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

def main(env, model, backbone='PointNet2', model_ver=-1, view_mode='overlap', policy='', gt_policy=False):
 
    env = gym.make(env, render_mode='human', sensor=PointCloudSensor, horizon=100, autoreset=True)
    scene_name = env.scene
    show_input, show_output, show_vis = False, True, True

    if model_ver > -1:
        model_dir = f'output/{scene_name}/{model}_{backbone}/version_{model_ver}/checkpoints/'
    else:
        model_dir = f'output/{scene_name}/{model}_{backbone}/'
        model_dir += sorted(map(lambda n: (len(n), n), os.listdir(model_dir)))[-1][1] # lastest version, sorted first by length and then by name
        model_dir += '/checkpoints/'
    model_dir += sorted(os.listdir(model_dir))[-1] # lastest checkpoint
    print('loading model', model_dir)

    # load model
    ae, open_dataset = create_model(model, backbone, scene_name, load_dir=model_dir)
    ae = ae.model.to(cfg.device)
    ae.eval()

    # load policy
    if policy:
        agent = MultiInputPolicy.load(policy)
    else:
        agent = None

    if model in ['Segmenter', 'GTSegmenter', 'MultiSegmenter']:
        classes = list(zip(env.classes, env.class_colors))
        C = len(classes)
        to_label = IntegerEncode(num_classes=C)
    if model == 'StatePredictor':
        from_state = StatePredictor.from_state(env)

    # Autoencoder Input Output
    def update_pc(env, robo_obs):
        nonlocal view_mode

        obs = env.observation

        preprocess = Normalize(obs['boundingbox'])
        postprocess = Unnormalize(obs['boundingbox'])
        orig = preprocess(obs_to_pc(obs, ['rgb']))

        target_pc, target_gts = orig, [preprocess(env.goal_state[env.goal_keys[0]].copy())]
        pred_pc, pred_feature, pred_gts = None, None, None

        if model == 'Autoencoder':
            # show the input and predicted pointclouds
            pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

            pred_pc = pred
            pred_feature = 'rgb'

        if model == 'Segmenter':
            pred = ae(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            pred = to_label(pred) # one hot to integer label (target is already integer label)

            pred_pc = pred
            pred_feature = 'seg'
            pred_gts = [mean_cube_pos(pred)]
        
        if model == 'MultiSegmenter':
            pred = ae.reconstruct_labeled(orig.to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

            pred_pc = pred
            pred_feature = 'seg'
            pred_gts = [mean_cube_pos(pred)]
        
        if model == 'StatePredictor':
            pred = ae(orig.to(cfg.device).unsqueeze(0))
            pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}

            pred_gts = [pred['cube_pos'], pred['robot0_eef_pos']]


        # assemble the pointclouds        
        if pred_pc is not None and pred_feature == 'seg': # convert to color
            pred_pc = np.concatenate((pred_pc[:, :3], seg_to_color(pred_pc[:, 3:], classes)), axis=1)

        # create visualization
        vis = np.array([]).reshape(0, 6)
        if target_gts is not None:
            for i, gt in enumerate(target_gts):
                vis = np.concatenate((vis, aa_lines(gt, np.array([0, 1, i*(len(target_gts)-1)]), res=50, length=0.1)), axis=0)
        if pred_gts is not None:
            for i, gt in enumerate(pred_gts):
                vis = np.concatenate((vis, aa_lines(gt, np.array([1, 0, i*(len(pred_gts)-1)]), res=50, length=0.1)), axis=0)

                
        target_pc = target_pc.cpu().numpy()
        
        # apply view mode
        if target_pc is not None and pred_pc is not None:
            if view_mode == 'sidebyside':
                # shift so they are next to each other
                pred_pc[:, 1] += 0.6

            if view_mode == 'overlap':
                # apply red tint and green tint
                target_pc[:, 3:] = interpolate_transition(target_pc[:, 3:], np.array([0, 1, 0]), 0.2)
                # pred_pc[:, 3:] = interpolate_transition(pred_pc[:, 3:], np.array([1, 0, 0]), 0.2)
            
        # merge input and visualizations
        pcs = []
        if show_input and target_pc is not None:
            pcs += [target_pc]
        if show_output and pred_pc is not None:
            pcs += [pred_pc]
        if show_vis and vis is not None:
            pcs += [vis]
        if len(pcs) > 0:
            pcs = postprocess(np.concatenate(tuple(pcs), axis=0))
            return pcs[:, :3], pcs[:, 3:]
        else:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    env.unwrapped.render_info = update_pc
    obs, info = env.reset()


    keep_running = True
    while keep_running:
        if agent:
            if gt_policy:
                gt_obs, gt_achieved = env.gt(env.observation)
                o = {
                    'observation': np.concatenate((env.proprioception, gt_obs), dtype=np.float32),
                    'achieved_goal': gt_achieved,
                    'desired_goal': env.gt.encode_goal(env.goal_state),
                }
            else:
                o = obs
            action, _states = agent.predict(o, deterministic=True)
        else:
            action = np.random.randn(env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        if env.viewer.is_pressed('g'): # show goal state
            env.show_frame(env.goal_state, None)

        if env.viewer.is_pressed('i'):
            show_input = not show_input
        if env.viewer.is_pressed('o'):
            show_output = not show_output
        if env.viewer.is_pressed('v'):
            show_vis = not show_vis



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--backbone', default='PointNet2', type=str)
    parser.add_argument('--model_ver', default=-1, type=int)
    parser.add_argument('--view', default='overlap', choices=['overlap', 'sidebyside'])
    parser.add_argument('--policy', default=None, type=str)
    parser.add_argument('--gt_policy', action='store_true')
    arg = parser.parse_args()

    main(arg.env, arg.model, arg.backbone, arg.model_ver, arg.view, arg.policy, arg.gt_policy)
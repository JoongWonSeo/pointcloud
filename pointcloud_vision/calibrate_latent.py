import numpy as np
import gymnasium as gym
import gymnasium_robotics
import robosuite_envs
from robosuite_envs.base_env import RobosuiteGoalEnv
import pointcloud_vision
from robosuite_envs.encoders import PassthroughEncoder
from pointcloud_vision.pc_sensor import PointCloudSensor
from pointcloud_vision.pc_encoder import GlobalSceneEncoder

from sb3_contrib.tqc.policies import MultiInputPolicy
from sb3_contrib import TQC

# This module suggests a latent space distance threshold by taking
# a successful pre-trained agent (usually trained on GT simulation),
# and then running it on a Vision simulation. The agent is then run
# until it reaches a success state, and the latent space distance is
# recorded, both right before it reaches the success state, and after.
# This is repeated for a number of runs, and the average of before success
# and after success is taken. A weighted average between the two is then
# taken, which is the suggested threshold.
# Therefore, we require:
# (1) a pre-trained agent to reach success,
# (2) a Vision simulation to record the latent space distance desired - acheived,
# (3) the ground-truth success checker to determine when the agent has reached success.


def latent_distributions(vision_task, policy_dir, horizon=50, runs=50, threshold_strictness=0.3, render=False, show_progress=False, save=True):
    env = gym.make(vision_task, render_mode='human' if render else None, max_episode_steps=horizon)
    # assert(isinstance(env, RobosuiteGoalEnv))
    gt_policy = MultiInputPolicy.load(policy_dir)

    if env.encoder.latent_threshold is None:
        print('latent_threshold is None, setting to 0')
        env.encoder.latent_threshold = np.zeros(env.encoder.get_goal_space(env.robo_env).shape)

    gt_encoder = PassthroughEncoder(env=env, obs_keys=env.encoder.obs_keys, goal_keys=env.encoder.goal_keys)

    # track stats for latent diff
    all_dists = []
    all_before_succ = []

    for i in range(runs):
        # reset env and get initial obs
        obs, info = env.reset()
        gt_goal = gt_encoder.encode_goal(env.goal_state)
        gt_obs, gt_achieved = gt_encoder(env.raw_state)
        success = env.check_success(gt_achieved, gt_goal, info=info, force_gt=True)
        if success:
            print('WARNING: success right after reset!')
            # continue
        dist = np.abs(env.goal_encoding - env.achieved)

        # per-episode stats
        zero_goal_space = np.zeros_like(env.goal_encoding)
        dist_sum = zero_goal_space.copy()
        dist_count = 0
        before_succ_sum = zero_goal_space.copy()
        before_succ_count = 0

        for t in range(horizon):
            # get actual (ground truth) observation and achieved goal
            gt = {
                'observation': np.concatenate((env.proprioception, gt_obs), dtype=np.float32),
                'achieved_goal': gt_achieved,
                'desired_goal': gt_goal,
            }
            # get action from policy
            action, _states = gt_policy.predict(gt, deterministic=True)

            # step env
            obs, reward, terminated, truncated, info = env.step(action)

            # determine latent space success
            gt_obs, gt_achieved = gt_encoder(env.observation)

            succ_prev = success # whether previous step was a success
            success = env.check_success(gt_achieved, gt_goal, info=info, force_gt=True)
            if success:
                if not succ_prev: # first success
                    before_succ_sum += dist
                    before_succ_count += 1
                dist = np.abs(env.goal_encoding - env.achieved)
                dist_sum += dist
                dist_count += 1

            if render:
                env.render()
            if show_progress:
                print(('#' * round((i*horizon+t)/(horizon*runs) * 100)).ljust(100, '-'), end='\r')

        # episode done
        if before_succ_count > 0:
            all_before_succ.append(before_succ_sum/before_succ_count)
        if dist_count > 0:
            all_dists.append(dist_sum/dist_count)
        else:
            print('WARNING: the policy failed in episode', i)


    if show_progress:
        print('\ndone')

    if len(all_before_succ) > 0:
        all_before_succ = np.stack(all_before_succ)
    if len(all_dists) > 0:
        all_dists = np.stack(all_dists)
    
    if len(all_before_succ) > 0 and len(all_dists) > 0:
        threshold = ((1-threshold_strictness) * all_before_succ.mean(axis=0) + threshold_strictness * (all_dists.mean(axis=0)))
    else:
        print('Warning: No data to calculate threshold')
        threshold = None

    if threshold is not None and save:
        env.encoder.save_latent_threshold(threshold, all_before_succ, all_dists)
    
    env.close()

    return threshold, all_before_succ, all_dists



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('vision_task', type=str)
    parser.add_argument('policy_dir', type=str)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--strictness', type=float, default=0.4)
    parser.add_argument('--show_distribution', action='store_true')
    parser.add_argument('--dont_save', action='store_true')
    arg = parser.parse_args()

    threshold, all_before_succ, all_dists = latent_distributions(arg.vision_task, arg.policy_dir, arg.horizon, arg.runs, arg.strictness, arg.render, show_progress=True, save=not arg.dont_save)

    print('Average latent space distance before success:')
    print(all_before_succ.mean(axis=0))
    print('Average latent space distance during success:')
    print(all_dists.mean(axis=0))

    print('Suggested latent space threshold:')
    print(threshold.__repr__())

    if arg.show_distribution:
        # show distribution of distances
        import matplotlib.pyplot as plt
        n = all_dists.shape[1]
        if n > 4:
            h = min(4, n) # max 4 plots per row
            w = int(np.ceil(n/h)) # ceil to get at least n plots
            fig, ax = plt.subplots(w, h, figsize=(h*4, w*4))

            for dim in range(all_before_succ.shape[1]):
                x, y = dim//h, dim%h
                ax[x, y].hist(all_before_succ[:,dim], bins=20, alpha=0.5, label='before success')
                ax[x, y].hist(all_dists[:,dim], bins=20, alpha=0.5, label='during success')
            lines, labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(lines, labels)
        else:
            fig, ax = plt.subplots(all_dists.shape[1])
            for dim in range(all_before_succ.shape[1]):
                ax[dim].hist(all_before_succ[:,dim], bins=20, alpha=0.5, label='before success')
                ax[dim].hist(all_dists[:,dim], bins=20, alpha=0.5, label='during success')
                # ax[dim].title('latent space dimension '+str(dim))
                ax[dim].legend()
        plt.show()


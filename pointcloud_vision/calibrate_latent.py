import numpy as np
import gymnasium as gym
import gymnasium_robotics
import robosuite_envs
import pointcloud_vision
from robosuite_envs.encoders import GroundTruthEncoder
from pointcloud_vision.pc_encoder import PointCloudSensor, PointCloudEncoder

from sb3_contrib.tqc.policies import MultiInputPolicy
from sb3_contrib import TQC

task = 'VisionReach-v0'
horizon = 50
runs = 100
render = False

env = gym.make(task, render_mode='human' if render else None, max_episode_steps=horizon)
# model = TQC.load('../rl/weights/'+task.replace('Vision', 'Robosuite'), env=env)
policy = MultiInputPolicy.load('../rl/weights/'+task.replace('Vision', 'Robosuite')+'_policy')

gt_encoder = GroundTruthEncoder(env=env, obs_keys=env.obs_encoder.obs_keys)



# run sim
obs, info = env.reset()
gt_goal = gt_encoder.encode(env.episode_goal_obs)
gt_eef = gt_encoder.encode(env.observation)
success = bool(np.linalg.norm(gt_goal-gt_eef) < 0.05)
succ_prev = success # whether previous step was a success
diff = np.abs(env.episode_goal_encoding - env.encoding)

# track stats for latent diff
sum_abs_diff = np.zeros_like(obs['desired_goal'])
count_abs_diff = np.zeros_like(obs['desired_goal'])
sum_before_succ = np.zeros_like(obs['desired_goal'])
count_before_succ = np.zeros_like(obs['desired_goal'])

total_abs_diff = np.zeros_like(obs['desired_goal'])
total_count_abs_diff = np.zeros_like(obs['desired_goal'])
total_before_succ = np.zeros_like(obs['desired_goal'])
total_count_before_succ = np.zeros_like(obs['desired_goal'])


for i in range(runs * horizon):
    gt_eef = gt_encoder.encode(env.observation)
    peg = {
            'observation': np.concatenate((env.proprioception, gt_eef), dtype=np.float32),
            'achieved_goal': gt_eef,
            'desired_goal': gt_goal,
        }
    action, _states = policy.predict(peg, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # determine latent space distance
    success = bool(np.linalg.norm(gt_goal-gt_eef) < 0.05)
    if success:
        if not succ_prev: # first success
            sum_before_succ += diff
            count_before_succ += 1
        diff = np.abs(env.episode_goal_encoding - env.encoding)
        sum_abs_diff += diff
        count_abs_diff += 1
    succ_prev = success

    env.render()

    if terminated or truncated:
        print('success' if info['is_success'] else 'failure')
        if count_before_succ.any():
            print('avg latent diff before success:', sum_before_succ/count_before_succ)
            total_before_succ += sum_before_succ/count_before_succ
            total_count_before_succ += 1
        if count_abs_diff.any():
            print('avg latent diff during success:', sum_abs_diff/count_abs_diff)
            total_abs_diff += sum_abs_diff/count_abs_diff
            total_count_abs_diff += 1
        obs, info = env.reset()
        gt_goal = gt_encoder.encode(env.episode_goal_obs)
        sum_abs_diff = np.zeros_like(obs['desired_goal'])
        count_abs_diff = np.zeros_like(obs['desired_goal'])
        sum_before_succ = np.zeros_like(obs['desired_goal'])
        count_before_succ = np.zeros_like(obs['desired_goal'])


print('------------------------')

if total_count_before_succ.any():
    print('avg latent diff before success:', total_before_succ/total_count_before_succ)
if total_count_abs_diff.any():
    print('avg latent diff during success:', total_abs_diff/total_count_abs_diff)

print('Suggested latent space threshold:')
print((0.5 * (total_before_succ/total_count_before_succ) + 0.5 * (total_abs_diff/total_count_abs_diff)).__repr__())
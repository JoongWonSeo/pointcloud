# show off all combinatations
import argparse
import numpy as np
import gymnasium as gym
from sb3_contrib.tqc.policies import MultiInputPolicy

from robosuite_envs import *
from pointcloud_vision import *

# parse arguments
sensors = {
    'default': None,
    'passthru': PassthroughSensor,
    'PC': PointCloudSensor,
}
encoders = {
    'default': None,
    'passthru': PassthroughEncoder,
    'AE': GlobalAEEncoder,
    'Seg': GlobalSegmenterEncoder,
    'MultiSeg': MultiSegmenterEncoder,
    'StatePred': StatePredictor,
    'StatePredVisGoal': StatePredictorVisualGoal,
}

parser = argparse.ArgumentParser()
parser.add_argument('env', type=str, help='environment ID')
parser.add_argument('--horizon', type=int, default=100, help='horizon')
parser.add_argument('--sensor', default='default', choices=list(sensors.keys()), help='sensor')
parser.add_argument('--encoder', default='default', choices=list(encoders.keys()), help='observation encoder')
parser.add_argument('--passive_encoder', default='', choices=list(encoders.keys()), help='passive encoder just for goal checking and visualization')
parser.add_argument('--policy', default='', type=str, help='path to policy file')
a = parser.parse_args()


# load environment
kwargs = {'sensor': sensors[a.sensor], 'encoder': encoders[a.encoder]}
if kwargs['encoder'] and kwargs['encoder'].requires_vision or a.passive_encoder and encoders[a.passive_encoder].requires_vision:
    kwargs['sensor'] = PointCloudSensor
env = gym.make(a.env, render_mode='human', max_episode_steps=a.horizon, **{k: v for k,v in kwargs.items() if v})

# create passive encoder
if a.passive_encoder and encoders[a.passive_encoder]:
    env.reset() # to get first obs
    pe = encoders[a.passive_encoder](env, env.obs_keys, env.goal_keys)
    if type(pe) is StatePredictor:
        pe.passthrough_goal = False
    pe_goal = pe.encode_goal(env.goal_obs)

    def show_sucess(h, w):
        # swap out the encoders temporarily
        env.unwrapped.encoder, orig = pe, env.encoder
        pe_achieved = pe.encode_goal(env.observation)
        pe_succ = env.check_success(pe_achieved, pe_goal, info=None)
        env.unwrapped.encoder = orig # restore original encoder

        overlay = np.zeros((h, w, 3))
        overlay[h-2:h, :, :] = [0, 1, 0] if pe_succ else [1, 0, 0]
        return overlay

    env.unwrapped.overlay = show_sucess
else:
    pe = None

# load policy
if a.policy:
    agent = MultiInputPolicy.load(a.policy)
else:
    agent = None

agent_input_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
agent_output_dim = env.action_space.shape[0]
assert all(-env.action_space.low == env.action_space.high)
agent_action_limit = env.action_space.high


# simulation
run = True
while run:
    obs, info = env.reset()

    if pe:
        pe_goal = pe.encode_goal(env.goal_obs)

    total_reward = 0
    for t in range(a.horizon):
        # select action
        if agent:
            action, _states = agent.predict(obs, deterministic=True)
        else:
            action = np.random.randn(agent_output_dim)

        # take action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # update results
        total_reward += reward            
        
        if env.viewer.is_pressed('g'): # show goal state
            env.show_frame(env.goal_state, None)
        if env.viewer.is_pressed('v'): # save visual goal 
            # pickle current robo obs
            import pickle
            with open(f'pointcloud_vision/input/{env.scene}/{a.env}_visual_goal.pkl', 'wb') as f:
                pickle.dump(env.raw_state, f)
                print('saved visual goal state')

        if terminated or truncated:
            break

    print(f"\ntotal_reward = {total_reward}\nis_success = {info['is_success']}")

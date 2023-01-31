import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from .base import RobosuiteGoalEnv, GroundTruthEncoder

class RobosuiteReach(RobosuiteGoalEnv):
    def __init__(self, render_mode=None):
        # create robosuite env
        robo_env = suite.make(
            env_name="Lift", # try with other tasks like "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            controller_configs=load_controller_config(default_controller="OSC_POSITION"),
            reward_shaping=False, # sparse reward
            ignore_done=True, # unlimited horizon (use gym's TimeLimit wrapper instead)
        )

        # define environment feedback functions
        def achieved_goal(state, proprioception):
            return proprioception # end-effector position
        
        def desired_goal(robo_obs):
            # goal is the cube position and end-effector position close to cube
            cube_pos = robo_obs['cube_pos']
            # add random noise to the cube position
            cube_pos[0] += np.random.uniform(-0.3, 0.3)
            cube_pos[1] += np.random.uniform(-0.3, 0.3)
            cube_pos[2] += np.random.uniform(0.01, 0.3)
            return cube_pos # end-effector should be close to the cube

        def check_success(achieved, desired, info):
            # batched version
            if achieved.ndim == 2:
                return np.linalg.norm(achieved - desired, axis=1) < 0.05
            else: # single version
                return np.linalg.norm(achieved - desired) < 0.05
        

        def render_goal(env, robo_obs):
            return np.array([env._episode_goal]), np.array([[1, 0, 0]])





        super().__init__(
            robo_env=robo_env,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            check_success=check_success,
            encoder=GroundTruthEncoder([], 'robot0_eef_pos'), # observation is only end-effector position
            render_mode=render_mode,
            render_info=render_goal
        )
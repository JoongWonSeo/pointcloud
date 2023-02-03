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
        def achieved_goal(proprioception, state):
            return proprioception # end-effector position
        
        def desired_goal(robo_obs):
            # goal is the cube position and end-effector position close to cube
            cube_pos = robo_obs['cube_pos']
            # add random noise to the cube position
            cube_pos[0] += np.random.uniform(-0.2, 0.2)
            cube_pos[1] += np.random.uniform(-0.2, 0.2)
            cube_pos[2] += np.random.uniform(0.01, 0.2)
            return cube_pos # end-effector should be close to the cube

        def check_success(achieved, desired, info):
            # batched version
            if achieved.ndim == 2:
                return np.linalg.norm(achieved - desired, axis=1) < 0.05
            else: # single version
                return np.linalg.norm(achieved - desired) < 0.05
        

        def render_goal(env, robo_obs):
            return np.array([env.episode_goal]), np.array([[1, 0, 0]])


        super().__init__(
            robo_env=robo_env,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            check_success=check_success,
            encoder=GroundTruthEncoder('robot0_eef_pos', []), # observation is only end-effector position
            render_mode=render_mode,
            render_info=render_goal
        )



class RobosuiteLift(RobosuiteGoalEnv):
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
        def achieved_goal(proprioception, state):
            return np.concatenate((proprioception, state)) # cube and end-effector position
        
        def desired_goal(robo_obs):
            # goal is the cube position and end-effector position close to cube
            cube_pos = robo_obs['cube_pos']
            # cube must be lifted up
            cube_pos[2] += 0.4
            # end-effector should be close to the cube
            eef_pos = cube_pos
            return np.concatenate((eef_pos, cube_pos))

        def check_success(achieved, desired, info):
            # batched version
            if achieved.ndim == 2:
                return np.linalg.norm(achieved - desired, axis=1) < 0.05
            else: # single version
                return np.linalg.norm(achieved - desired) < 0.05
        

        def render_goal(env, robo_obs):
            eef_goal = env.episode_goal[:3]
            cube_goal = env.episode_goal[3:]
            return np.array([eef_goal, cube_goal]), np.array([[1, 0, 0], [0, 1, 0]])


        super().__init__(
            robo_env=robo_env,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            check_success=check_success,
            encoder=GroundTruthEncoder('robot0_eef_pos', 'cube_pos'), # observation is end-effector position and cube position
            render_mode=render_mode,
            render_info=render_goal
        )


class RobosuitePickAndPlace(RobosuiteGoalEnv):
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
        def achieved_goal(proprioception, state):
            return state # cube position
        
        def desired_goal(robo_obs):
            # goal is the cube position and end-effector position close to cube
            cube_pos = robo_obs['cube_pos']
            # cube must be moved
            cube_pos[0] += np.random.uniform(-0.2, 0.2)
            cube_pos[1] += np.random.uniform(-0.2, 0.2)
            if np.random.uniform() < 0.5: # cube in the air for 50% of the time
                cube_pos[2] += np.random.uniform(0.01, 0.2)
            return cube_pos

        def check_success(achieved, desired, info):
            # batched version
            if achieved.ndim == 2:
                return np.linalg.norm(achieved - desired, axis=1) < 0.05
            else: # single version
                return np.linalg.norm(achieved - desired) < 0.05
        

        def render_goal(env, robo_obs):
            cube_goal = env.episode_goal
            return np.array([cube_goal]), np.array([[1, 0, 0]])


        super().__init__(
            robo_env=robo_env,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            check_success=check_success,
            encoder=GroundTruthEncoder('robot0_eef_pos', 'cube_pos'), # observation is end-effector position and cube position
            render_mode=render_mode,
            render_info=render_goal
        )
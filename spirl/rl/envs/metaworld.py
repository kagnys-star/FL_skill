import numpy as np
from collections import defaultdict
import metaworld
import random
import torch
from spirl.utils.pytorch_utils import ar2ten, ten2ar
from gymnasium.spaces import Box
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv

TASKS = ['reach-v2','door-open-v2','drawer-open-v2','button-press-v2', "push-v2" , 'drawer-close-v2']

def goal_hand_generation():
    # 손의 X축 범위와 간격 계산
    num_hands = 4
    x_hand_min, x_hand_max = -0.525, 0.525
    y_hand_value = 0.6  
    z_hand_value = 0.2  # Z축 고정값 (높이)

    # X축에서 동일 간격으로 손의 위치 생성
    x_hand_positions = np.linspace(x_hand_min, x_hand_max, num_hands)
    initial_hand_positions = np.array([[x, y_hand_value, z_hand_value] for x in x_hand_positions])

    # Goal Space에서 목표의 X축 범위와 간격 계산
    x_goal_min, x_goal_max = -0.1, 0.1
    y_goal_value = 0.9  # Y축 고정값 (Goal Space 범위 내 중간)
    z_goal_value = 0.3   # Z축 고정값

    # X축에서 동일 간격으로 목표 위치 생성 (손의 위치와 비슷한 간격으로 배치)
    x_goal_positions = np.linspace(x_goal_min, x_goal_max, num_hands)
    goal_positions = np.array([[x, y_goal_value, z_goal_value] for x in x_goal_positions])
    
    return initial_hand_positions, goal_positions

def same_position_generations(obj_low,obj_high,num_goals):
    # 손의 X축 범위와 간격 계산
    y_hand_value = obj_low[1] + (obj_high[1]-obj_low[1])/2  
    z_hand_value = obj_high[2]  # Z축 고정값 (높이)

    # X축에서 동일 간격으로 손의 위치 생성
    x_hand_positions = np.linspace(obj_low[0], obj_high[0], num_goals)
    initial_hand_positions = np.array([[x, y_hand_value, z_hand_value] for x in x_hand_positions])
    return initial_hand_positions


RAND_VEC =[np.array([-0.1, 0.9, 0.0]),np.array([-0.05, 0.9, 0.0]),np.array([0.05, 0.9, 0.0]),np.array([0.1, 0.9, 0.0])]



class FIXEDMT10(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.task_id = TASKS[self._hp.task_id]
        self._env = self._make_env(seed = self._hp.seed)

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "MT-10_reach-v2",
            'reward_norm': 1.,
        }))

    def _make_env(self, seed):
        from gym import wrappers
        ml1 = metaworld.ML1(self.task_id)
        env = ml1.train_classes[self.task_id]()
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        env._partially_observable = False
        env.random_init = False
        if "drawer-open" in self.task_id: env._last_rand_vec = np.concatenate((env.init_config["obj_init_pos"],env.init_config['obj_init_pos'] + np.array([.0, -.16 - env.maxDist, .09])))
        elif "drawer-close" in self.task_id: env._last_rand_vec = np.concatenate((env.init_config["obj_init_pos"],env.init_config['obj_init_pos'] +  np.array([.0, -.16, .09])))
        elif "door" in self.task_id: env._last_rand_vec = np.concatenate((env.init_config["obj_init_pos"],env.init_config['obj_init_pos'] + np.array([-0.3, -0.45, 0.])))
        else: env._last_rand_vec = np.concatenate((env.init_config["obj_init_pos"],env.goal))
        env._freeze_rand_vec = True
        env.seed(seed)
        env.reset()
        return env
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        if info["success"] == 1:
            self.sucess_info = 1.
            done = True
        return self._wrap_observation(obs),  np.float64(self.sucess_info), np.array(done), info
    
    def reset(self):
        obs = self._env.reset()
        self.sucess_info = 0.
        return self._wrap_observation(obs)

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(success=self.sucess_info))
        return info

class MIX_TASK(FIXEDMT10):

    def _make_env(self, seed):
        from gym import wrappers
        if self.task_id < 4:
            taks = "reach-v2"
        elif self.task_id < 8:
            taks = "drawer-open-v2"
        else:
            taks = "door-open-v2"
        ml1 = metaworld.ML1(taks)
        env = ml1.train_classes[taks]()
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        env._partially_observable = False
        env.random_init = False
        if self.task_id < 4:
            initial_hand_positions , goal_positions = goal_hand_generation()
            env.hand_init_pos = initial_hand_positions[self.task_id]
            env._last_rand_vec = np.concatenate((env.init_config["obj_init_pos"],goal_positions[self.task_id]))
            env.goal = goal_positions[self.task_id]
        elif self.task_id < 8:
            idx = self.task_id - 4
            env.init_config['obj_init_pos'] = RAND_VEC[idx]
            env._last_rand_vec = np.concatenate((env.init_config["obj_init_pos"], env.init_config["obj_init_pos"] + np.array([.0, -.16 - env.maxDist, .09])))
        else:
            idx = self.task_id - 8
            low_bound = env._random_reset_space.low
            high_bound = env._random_reset_space.high
            obj = same_position_generations(low_bound[:3],high_bound[:3],4)
            goal = np.array([ x + np.array([-0.3, -0.45, 0.]) for x in obj])
            env._last_rand_vec = np.concatenate((obj[idx],goal[idx]))
            env.init_config['obj_init_pos'] = obj[idx]
            env.goal = goal[idx]
        env._freeze_rand_vec = True
        env.seed(seed)
        env.reset()
        return env
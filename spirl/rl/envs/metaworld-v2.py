import numpy as np
from collections import defaultdict
import metaworld
import random
import torch
from spirl.utils.pytorch_utils import ar2ten, ten2ar
from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDrawerOpenEnvV2
from gymnasium.spaces import Box
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv

def goal_hand_generation():
    # 손의 움직임 범위 설정 (_HAND_SPACE)
    hand_space = Box(np.array([-0.525, 0.348, -0.0525]), np.array([0.525, 1.025, 0.7]), dtype=np.float64)

    # 목표의 범위 설정 (goal box)
    goal_space = Box(np.array([-0.1, 0.8, 0.05]), np.array([0.1, 0.9, 0.3]), dtype=np.float64)

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

RAND_VEC =[np.array([-0.1, 0.9, 0.0]),np.array([-0.05, 0.9, 0.0]),np.array([0.05, 0.9, 0.0]),np.array([0.1, 0.9, 0.0])]

class FixedDrawerOpenEnv(SawyerDrawerOpenEnvV2):
    def __init__(self):
        super().__init__()
        self._random_reset_space = None
        self._partially_observable = False
        # 커스터마이징: 새로운 보상이나 목표 설정
    
    def set_fix_obj_goal(self,rand_vec):
        self._set_task_called = True
        self._last_rand_vec = rand_vec
        self._freeze_rand_vec = True

class MT1(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.task_id = self._hp.task_id
        self.initial_hand_positions , self.goal_positions = goal_hand_generation()
        self._env = self._make_env(self.task_id, seed = self._hp.seed)

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "MT-1_reach",
            'reward_norm': 1.,
        }))

    def _make_env(self, task_id, seed):
        import gym
        from gym import wrappers
        ml1 = metaworld.ML1('reach-v2')
        env = ml1.train_classes['reach-v2']()
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        env._partially_observable = False
        env.hand_init_pos = self.initial_hand_positions[task_id]
        rand_vec = env._last_rand_vec
        new_rand_vec = np.concatenate((env.init_config["obj_init_pos"],self.goal_positions[task_id]))
        env._last_rand_vec = new_rand_vec
        env.seed(seed)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env
    
    def step(self, action):
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            obs, reward, trunc, termn, info = self._env.step(action)
            done = trunc or termn
            reward = reward / self._hp.reward_norm
        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}
        if info["success"] == 1:
            self.sucess_info = 1.
        return self._wrap_observation(obs),  np.float64(reward), np.array(done), info
    
    def reset(self):
        obs, info = self._env.reset()
        self.sucess_info = 0.
        return self._wrap_observation(obs)

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(success=self.sucess_info))
        return info
    

class drawer_open(MT1):

    def _make_env(self, task_id, seed):
        from gym import wrappers
        env = FixedDrawerOpenEnv()
        env.set_fix_obj_goal(RAND_VEC[task_id])
        env.seed(seed)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env
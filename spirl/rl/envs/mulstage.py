import numpy as np
from collections import defaultdict
import random
import torch
from spirl.utils.pytorch_utils import ar2ten, ten2ar
from gymnasium.spaces import Box
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv
from metaworld_complex.fed_env import MetaWorldComplexEnv
from metaworld_complex.envs.sawyer_complex_v2 import SawyerComplexEnvV2

TASKS = [ ["box", "drawer"] , ["box", "button"], ["box", "door"]]

class mulstage(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.task_id = TASKS[self._hp.task_id]
        self._env = self._make_env(self.task_id, seed = self._hp.seed)

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "mul_stage",
            'reward_norm': 1.,
        }))

    def _make_env(self, task_id, seed):
        env_cls = "MC4"
        task = ['box', 'drawer', 'button', 'door']
        benchmark = SawyerComplexEnvV2
        env = benchmark(task)
        env = MetaWorldComplexEnv(env_cls, env, task, task_id, seed=seed)
        return env
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return self._wrap_observation(obs),  np.float64(rew), np.array(done), info
    
    def reset(self):
        obs = self._env.reset()
        return self._wrap_observation(obs)

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(success=self._env.success))
        return info


class Lmulstage(mulstage):

    def _make_env(self, task_id, seed):
        env_cls = "MC4"
        task = ['box', 'drawer', 'button', 'door']
        benchmark = SawyerComplexEnvV2
        env = benchmark(task)
        env = MetaWorldComplexEnv(env_cls, env, task, task, seed=seed)
        return env
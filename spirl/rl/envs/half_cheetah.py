import numpy as np
from collections import defaultdict
import d4rl

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class cheetah(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.task_id = self._hp.task_id
        self._env = self._make_env(self._hp.name, self.task_id)

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "HalfCheetah-v3",
            'reward_norm': 1.,
        }))
    def _wrap_observation(self, obs):
        one_hot = np.zeros(4)
        one_hot[self.task_id] = 1
        new_obs = np.concatenate([obs,one_hot])
        return np.asarray(new_obs, dtype=np.float32)
    
    def _make_env(self, id,task):
        import gym
        from gym import wrappers
        env = gym.make(id , xml_file = f"/home/kangys/workspace/cheetah_FL/half_cheetahs/{task + 1}.xml")
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, info     # casting reward to float64 is important for getting shape later

    def reset(self):
        return super().reset()
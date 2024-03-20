import numpy as np
from collections import defaultdict
import metaworld
import random
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class MetaworldEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(id = self._hp.name,seed = self._hp.seed)
        self.n_total_tasks = self._hp.n_total_tasks
        self.task_index = self._hp.task_index


    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        if info["success"] == 1:
            done = np.array(True)
        return obs, np.float64(rew), done, info    # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    def _wrap_observation(self, obs):
        one_hot = np.zeros(self.n_total_tasks)
        one_hot[self.task_index] = 1.0
        return np.concatenate([obs, one_hot])

    def _make_env(self, id, seed):
        """Instantiates the environment given the ID."""
        import gym
        from gym import wrappers
        #meta env change these part
        MT10 = metaworld.MT10(seed=seed)
        env_cls = MT10.train_classes[id]
        env = env_cls()
        task = random.choice([task for task in MT10.train_tasks
                        if task.env_name == id])
        env.set_task(task)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env
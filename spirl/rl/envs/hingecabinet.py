import numpy as np
from collections import defaultdict
import d4rl

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class KitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']
    
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self._env.init_qpos = np.array([-2.0077081e+00, -1.4741026e+00,  1.0114931e+00, -1.9735739e+00,
        2.3208299e-01,  1.8048230e+00,  1.1531316e+00,  4.1284524e-02,
        1.6143329e-02,  3.2037016e-04,  2.8416052e-04,  3.5455654e-05,
        2.0940402e-05,  3.7188751e-05,  1.8810410e-06, -1.2032897e-05,
       -2.8809134e-05,  6.1661260e-05, -2.0045940e-04,  1.2853415e-01,
       -2.5915236e-03, -1.3205803e-03, -7.6801747e-01, -2.3493095e-01,
        7.2973716e-01,  1.6189475e+00,  1.0021502e+00,  7.6331699e-04,
       -5.5140192e-03, -6.6658981e-02])
        self._env.TASK_ELEMENTS = ['hinge cabinet']

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "kitchen-mixed-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        completed_subtasks = info.pop("completed_tasks")
        for task in self.SUBTASKS:
            self.solved_subtasks[task] = 1 if task in completed_subtasks or self.solved_subtasks[task] else 0
        return info


class NoGoalKitchenEnv(KitchenEnv):
    """Splits off goal from obs."""
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        obs = obs[:int(obs.shape[0]/2)]
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:int(obs.shape[0]/2)]

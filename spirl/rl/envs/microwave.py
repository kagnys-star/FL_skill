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
        self._env.init_qpos = np.array([-1.3389963e+00, -1.3264140e+00,  1.3405436e+00, -2.2044790e+00,
        5.3182673e-03,  1.8399799e+00,  2.1059594e+00,  2.0476280e-02,
        2.7922874e-02, -1.4271861e-04, -3.1779910e-04,  5.4859749e-05,
       -3.0485082e-05,  2.2279353e-05, -4.7772624e-05, -7.6423168e-01,
       -4.3449299e-03,  9.0664180e-05, -1.7343283e-04,  3.9343673e-04,
        4.6323407e-03, -9.0524862e-03, -7.7819896e-01, -2.6868576e-01,
        3.5023639e-01,  1.6193352e+00,  9.9466020e-01, -6.7910752e-03,
       -1.5635670e-03, -2.8064189e-04])
        self._env.TASK_ELEMENTS = ['microwave'] 
        
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

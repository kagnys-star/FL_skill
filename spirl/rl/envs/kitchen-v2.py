import numpy as np
from collections import defaultdict
import d4rl

import random
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class KitchenEnv_v2(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    
    SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']

    INIT_POS = {
        "bottom burner" : np.array([-1.0964799e+00, -1.7721732e+00,  1.8203166e+00, -2.2051444e+00,
       -4.2435598e-01,  1.3044175e+00,  2.2836616e+00,  3.1263143e-02,
        2.8937008e-02,  3.8834161e-04,  1.0496432e-04,  7.3471597e-06,
       -4.2346783e-05, -1.0638056e-05,  2.7669792e-05,  6.6587105e-05,
        1.0965980e-05, -4.2847682e-06, -4.0668776e-04, -1.2338620e-04,
        9.3456917e-03,  1.2382864e-03, -7.0865969e-03, -1.9472559e-01,
        5.5426806e-01,  1.7533431e+00,  9.8377204e-01,  1.1427832e-01,
       -2.8078513e-02, -1.3373458e-01]),
       'hinge cabinet' : np.array([-2.0077081e+00, -1.4741026e+00,  1.0114931e+00, -1.9735739e+00,
        2.3208299e-01,  1.8048230e+00,  1.1531316e+00,  4.1284524e-02,
        1.6143329e-02,  3.2037016e-04,  2.8416052e-04,  3.5455654e-05,
        2.0940402e-05,  3.7188751e-05,  1.8810410e-06, -1.2032897e-05,
       -2.8809134e-05,  6.1661260e-05, -2.0045940e-04,  1.2853415e-01,
       -2.5915236e-03, -1.3205803e-03, -7.6801747e-01, -2.3493095e-01,
        7.2973716e-01,  1.6189475e+00,  1.0021502e+00,  7.6331699e-04,
       -5.5140192e-03, -6.6658981e-02]),
       'light switch' : np.array([-1.3389963e+00, -1.3264140e+00,  1.3405436e+00, -2.2044790e+00,
        5.3182673e-03,  1.8399799e+00,  2.1059594e+00,  2.0476280e-02,
        2.7922874e-02, -1.4271861e-04, -3.1779910e-04,  5.4859749e-05,
       -3.0485082e-05,  2.2279353e-05, -4.7772624e-05, -7.6423168e-01,
       -4.3449299e-03,  9.0664180e-05, -1.7343283e-04,  3.9343673e-04,
        4.6323407e-03, -9.0524862e-03, -7.7819896e-01, -2.6868576e-01,
        3.5023639e-01,  1.6193352e+00,  9.9466020e-01, -6.7910752e-03,
       -1.5635670e-03, -2.8064189e-04]),
        'microwave' : np.array([-1.3389963e+00, -1.3264140e+00,  1.3405436e+00, -2.2044790e+00,
        5.3182673e-03,  1.8399799e+00,  2.1059594e+00,  2.0476280e-02,
        2.7922874e-02, -1.4271861e-04, -3.1779910e-04,  5.4859749e-05,
       -3.0485082e-05,  2.2279353e-05, -4.7772624e-05, -7.6423168e-01,
       -4.3449299e-03,  9.0664180e-05, -1.7343283e-04,  3.9343673e-04,
        4.6323407e-03, -9.0524862e-03, -7.7819896e-01, -2.6868576e-01,
        3.5023639e-01,  1.6193352e+00,  9.9466020e-01, -6.7910752e-03,
       -1.5635670e-03, -2.8064189e-04]),
       'slide cabinet' : np.array([-1.2606528e+00, -1.5796804e+00,  1.4097186e+00, -1.9951299e+00,
        2.8924993e-01,  1.6060121e+00,  1.4530079e+00,  4.6622805e-02,
       -6.9765688e-04, -2.8959336e-04, -4.4677241e-04,  4.1309766e-05,
       -3.1401258e-05,  5.3994063e-05, -4.7984020e-05, -8.7706196e-01,
       -5.0564501e-03, -4.5872766e-01, -3.3244222e-02,  3.9439337e-04,
        3.6373355e-03,  6.9298348e-03,  6.9463653e-03, -2.1048030e-01,
        7.5253046e-01,  1.6189740e+00,  9.9168390e-01, -1.6366837e-03,
       -1.2457009e-03,  8.8523656e-02]),
       'top burner' : np.array([-1.2429860e+00, -1.2411315e+00,  1.4726796e+00, -2.3160005e+00,
        1.2739329e-01,  1.7178695e+00,  2.0256391e+00, -2.7657372e-03,
        3.4838781e-02,  4.2189559e-04,  9.4233452e-05, -6.4548671e-01,
       -3.6665087e-03,  6.6202396e-05, -2.7958586e-05,  5.6617569e-05,
       -4.7323683e-05,  4.2330059e-05, -3.7956994e-04, -7.0084090e-05,
       -1.5387173e-03, -3.2628333e-04, -7.4980235e-01, -2.6945311e-01,
        3.4959471e-01,  1.6189555e+00,  1.0086857e+00, -2.7244345e-03,
        4.4570896e-03, -4.5230633e-04])
    }

    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        if self._hp.tasks in self.INIT_POS.keys():
            self._env.init_qpos = self.INIT_POS[self._hp.tasks]
        self._env.ENFORCE_TASK_ORDER = False
        self._env.TASK_ELEMENTS = [self._hp.tasks]
        self._env.tasks_to_complete = list(self._env.TASK_ELEMENTS)


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


class NoGoalKitchenEnv(KitchenEnv_v2):
    """Splits off goal from obs."""
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        obs = obs[:int(obs.shape[0]/2)]
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:int(obs.shape[0]/2)]
    
class RadomKitchenEnv(KitchenEnv_v2):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self.task_num = 0
        self.choose_env()
    
    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        self.choose_env()
        return super().reset()

    def choose_env(self):
        self.task_num = random.randrange(0,7)
        self._env.TASK_ELEMENTS = [self.SUBTASKS[self.task_num]]
        self._env.init_qpos = self.INIT_POS[self.SUBTASKS[self.task_num]]
        self._env.ENFORCE_TASK_ORDER = False
        self._env.tasks_to_complete = list(self._env.TASK_ELEMENTS)
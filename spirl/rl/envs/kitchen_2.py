import numpy as np
from collections import defaultdict
import random
import d4rl

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class KitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    
    SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']
    INIT_STATES = [np.array([ 1.48388023e-01, -1.76848573e+00,  1.84390296e+00, -2.47685760e+00,
                                    2.60252026e-01,  7.12533105e-01,  1.59515394e+00,  4.79267505e-02,
                                    3.71350919e-02, -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
                                   -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
                                    4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
                                   -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
                                    3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
                                   -6.62095997e-03, -2.68278933e-04]), np.array([-7.4123758e-01, -1.7546432e+00,  1.9558041e+00, -1.6154059e+00,
       -5.4668921e-01,  1.2045333e+00,  2.4618077e+00,  2.5471697e-02,
        1.9734293e-02, -7.9886944e-05, -4.2385687e-04,  1.9689563e-05,
        3.9980623e-05,  6.9879199e-05,  2.0673247e-05, -1.9986381e-05,
        2.8734530e-05, -3.9337866e-04,  2.5174528e-04,  1.2424939e-05,
        5.7478216e-03,  5.6894189e-03, -5.8231127e-01, -2.6878706e-01,
        3.4955165e-01,  1.6189430e+00,  1.0019971e+00, -7.0487643e-03,
       -2.1925075e-03, -4.1788328e-04]), np.array([-1.2674966e+00, -1.5856974e+00,  1.4021021e+00, -1.9417316e+00,
        3.2742313e-01,  1.5248332e+00,  1.3944041e+00,  4.6823174e-02,
       -4.9968548e-03, -3.4511494e-04,  2.8893608e-04,  4.6096657e-06,
       -1.0731519e-06,  6.9477443e-05,  3.1747608e-05, -8.7703949e-01,
       -5.0776331e-03, -5.5758768e-01, -4.0247560e-02,  3.0837418e-04,
        5.3328201e-03,  6.5747132e-03,  3.2931585e-03, -2.1026017e-01,
        7.5243098e-01,  1.6189126e+00,  1.0055639e+00,  1.8677633e-03,
       -2.8992987e-03,  8.8044100e-02]), np.array([-2.2246664e+00, -1.4129806e+00,  1.0475178e+00, -1.9299496e+00,
        2.3793522e-01,  1.8324754e+00,  1.1345068e+00,  3.4153547e-02,
        1.6451010e-04,  5.1391480e-04, -3.3406733e-04,  5.6890520e-05,
        3.3273012e-05,  1.5545613e-05, -4.3130811e-05,  5.4660508e-05,
        8.4606927e-06, -6.3296378e-05,  1.7659320e-04,  2.6348272e-01,
        2.8637601e-03,  3.0351961e-03, -7.7633512e-01, -2.3456331e-01,
        7.2968483e-01,  1.6189239e+00,  1.0036634e+00, -6.5602884e-03,
       -9.3019847e-03, -6.6156141e-02]), np.array([-1.2150208e+00, -1.7728019e+00,  1.8986284e+00, -2.0606890e+00,
       -4.2002100e-01,  1.3374336e+00,  2.3152425e+00,  2.7378878e-02,
        2.5079377e-02,  5.8420639e-05,  9.7897602e-05, -1.2832818e-06,
        4.3017291e-05, -1.4175036e-05, -4.9132534e-05,  5.2295291e-05,
       -1.6413776e-05, -2.7060023e-04,  1.8146726e-04, -4.3322358e-04,
       -4.7870786e-03, -7.6087657e-03, -6.9890708e-01, -2.2289988e-01,
        5.7098156e-01,  1.6762449e+00,  9.7355604e-01,  1.1408262e-01,
        4.6889331e-02, -1.6854040e-01]),np.array([-1.2935489e+00, -1.2578226e+00,  1.3557991e+00, -2.2373316e+00,
        9.9678762e-02,  1.8058912e+00,  2.1831472e+00,  1.9958969e-03,
        3.7682928e-02,  6.9472124e-05,  1.5574187e-04,  6.4468877e-05,
       -1.9493506e-05, -1.1361010e-05,  6.3432879e-08, -8.1335491e-01,
       -4.6323389e-03, -4.3606698e-05, -6.2289029e-05,  3.9835775e-04,
        4.9169930e-03,  8.3130744e-04, -7.8612494e-01, -2.6859683e-01,
        3.5014573e-01,  1.6188477e+00,  9.9919045e-01, -5.6040259e-03,
       -1.2207832e-04, -4.4425117e-04]),np.array([-1.1859719e+00, -1.1959008e+00,  1.4937971e+00, -2.3043234e+00,
        3.3233479e-01,  1.6422106e+00,  1.9930224e+00, -4.4310484e-03,
        3.2830045e-02, -1.1992746e-04, -3.9161413e-04, -7.9655397e-01,
       -4.5248806e-03,  1.2208996e-05,  1.4015675e-05, -7.4952845e-06,
        2.8450329e-05,  3.1551515e-04,  2.2501154e-05,  4.6929909e-04,
       -6.6485293e-03, -1.5214891e-03, -7.4345136e-01, -2.6946035e-01,
        3.4993225e-01,  1.6190597e+00,  9.9773061e-01,  8.3500650e-03,
        9.8998973e-04, -1.6257125e-04])]

    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self.task_num = 0
        self.choose_env()

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "kitchen-mixed-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        self.choose_env()
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
    

    def choose_env(self):
        self.task_num = random.randrange(0,7)
        self._env.TASK_ELEMENTS = [self.SUBTASKS[self.task_num]]
        self._env.init_qpos = self.INIT_STATES[self.task_num]

 


class NoGoalKitchenEnv(KitchenEnv):
    """Splits off goal from obs."""
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        obs = obs[:int(obs.shape[0]/2)]
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:int(obs.shape[0]/2)]

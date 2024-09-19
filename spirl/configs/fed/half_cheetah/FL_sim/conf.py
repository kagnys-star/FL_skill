import os

from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl, Prox_clients
from spirl.components.logger import Logger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.half_cheetah import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator ,SequenceEvaluator
current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ClSPiRLMdl,
    'logger': Logger,
    'data_dir': "../data/cheetah/1",
    'epoch_cycles_train': 50,
    'num_epochs': 5,
    'evaluator': SequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
    'batch_size': 128,
    'optimizer': 'asam',
    'lr': 1e-3,
    'gradient_clip': None,
    'init_grad_clip': 0.001,
    'init_grad_clip_step': 100,     # clip gradients in initial N steps to avoid NaNs
    'momentum': 0,      # momentum in RMSProp / SGD optimizer
    'adam_beta': 0.9,       # beta1 param in Adam
    'top_of_n_eval': 1,     # number of samples used at eval time
    'top_comp_metric': None,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped

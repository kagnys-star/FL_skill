import flwr as fl
import os
from flwr.server.strategy import FedAvg , FedAdam
import math
from strategy import *
from spirl.components.params import get_args
import matplotlib; matplotlib.use('Agg')
import torch
import os
import time
from shutil import copy
import datetime
import imp
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch import autograd
from spirl.modules.losses import L2Loss
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict , AttrDict, ParamDict
from spirl.components.checkpointer import get_config_path
from spirl.utils.pytorch_utils import  ten2ar
from spirl.components.trainer_base import BaseTrainer
from spirl.utils.wandb import WandBLogger
from collections import OrderedDict

import wandb
WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'


class ServerModel(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.setup_device()
        # set up params
        self.conf = conf = self.get_config()
        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
        self.conf = self.postprocess_conf(conf)
        if args.deterministic: set_seeds()
        # set up logging + training monitoring
        self.writer = self.setup_logging(conf, self.log_dir)

        # buld dataset, model. logger, etc.
        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=1,
                                dataset_size=-1)
        self.logger, self.model, self.train_loader = self.build_phase(train_params, 'train')

        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger)

    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,
            'logger': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 32,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'minimizer ' : None,
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
            'logging_target': 'wandb',
        })
        return default_dict

    def val(self , rounds):
        print('server Testing')
        start = time.time()
        losses_meter = RecursiveAverageMeter()
        iw_meter = RecursiveAverageMeter()
        z_samples  = []
        z_q_samples  = []
        log_q_zx_samples = []
        log_q_z_qx_samples = []
        mu_samples = []
        self.model.eval()
        self.evaluator.reset()
        with autograd.no_grad():
            for sample_batched in self.train_loader:
                
                inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

                # run evaluator with val-mode model
                with self.model.val_mode():
                    self.evaluator.eval(inputs, self.model)

                # run non-val-mode model (inference) to check overfitting
                output = self.model(inputs)
                losses = self.model.loss(output, inputs)
                output.q_reconstruction = self.model.decode(output.z_p,
                                            cond_inputs=self.model._learned_prior_input(inputs),
                                            steps=self.model._hp.n_rollout_steps,
                                            inputs=inputs)
                losses.prior_mse = L2Loss(1.0)(inputs.actions,(output.q_reconstruction))
                iw_meter.update(AttrDict(LL = self.importants_weights(output, inputs)))
                losses_meter.update(losses)
                log_q_zx_samples.append(ten2ar(output.q.log_prob(output.z)))
                log_q_z_qx_samples.append(ten2ar(output.q_hat.log_prob(output.z_p)))
                z_samples.append(ten2ar((-0.5 *(output.z ** 2) + math.log(math.sqrt(2*math.pi))).sum(dim=1)))
                z_q_samples.append(ten2ar((-0.5 *(output.z_p ** 2) + math.log(math.sqrt(2*math.pi))).sum(dim=1)))
                mu_samples.append(ten2ar(output.q.mu))
                del losses
            

            if self.evaluator is not None:
                self.evaluator.dump_results(rounds)
            log_q_z = np.concatenate(np.array(z_samples), axis=0)
            log_q_z_q = np.concatenate(np.array(z_q_samples), axis=0)
            #q(z|x)
            log_q_zx = np.concatenate(np.array(log_q_zx_samples), axis=0)
            log_q_z_qx = np.concatenate(np.array(log_q_z_qx_samples), axis=0)
            #p(z)
            mi_estimate = (log_q_zx - log_q_z)
            mi_estimate2 = (log_q_z_qx - log_q_z_q)
            mu_s = np.concatenate(np.array(mu_samples), axis=0)
            z_var = np.var(mu_s, axis=0, ddof=1)
            threshold = 0.01
            activated_units = (z_var > threshold).astype(float)
            self.logger.log_scalar(np.mean(activated_units), "AU", rounds, "val")
            #self.logger.log_scalar(np.mean(log_q_z), "log_q_z", rounds, "val")
            #self.logger.log_scalar(np.mean(log_q_zx), "log_q_zx", rounds, "val")
            self.logger.log_scalar(np.mean(mi_estimate), "mi_estimate", rounds, "val")
            self.logger.log_scalar(np.mean(mi_estimate2), "prior_mi_estimate", rounds, "val")
            self.logger.log_scalar_dict(iw_meter.avg ,'val', rounds )
            self.model.log_outputs(output, inputs, losses_meter.avg, rounds,
                                        log_images=False, phase='val', **self._logging_kwargs)
            print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                    .format(losses_meter.avg.total.value.item(), time.time() - start)))
        del output , inputs
        return losses_meter.avg.total.value.item()


    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and model configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration
        conf.model = conf_module.model_config

        # data config
        try:
            data_conf = conf_module.data_config
        except AttributeError:
            data_conf_file = imp.load_source('dataset_spec', os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
            data_conf = AttrDict()
            data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
            data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
        conf.data = data_conf

        # model loading config
        conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

        return conf

    def postprocess_conf(self, conf):
        conf.model['batch_size'] = self._hp.batch_size if not torch.cuda.is_available() \
            else int(self._hp.batch_size / torch.cuda.device_count())
        conf.model.update(conf.data.dataset_spec)
        conf.model['device'] = conf.data['device'] = self.device.type
        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            if self._hp.logging_target == 'wandb':
                exp_name = f"{'_'.join(self.args.path.split('/')[-3:])}_{self.args.prefix}" if self.args.prefix \
                    else os.path.basename(self.args.path)
                writer = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                     path=self._hp.exp_path, conf=conf, exclude=['model_rewards', 'data_dataset_spec_rewards'])
            else:
                writer = SummaryWriter(log_dir)
        else:
            writer = None

        # set up additional logging args
        self._logging_kwargs = AttrDict(
        )
        return writer


    def build_phase(self, params, phase):
        if not self.args.dont_save:
            if self._hp.logging_target == 'wandb':
                logger = self.writer
            else:
                logger = params.logger_class(self.log_dir, summary_writer=self.writer)
        else:
            logger = None
        model = params.model_class(self.conf.model, logger)
        if torch.cuda.device_count() > 1:
            raise ValueError("Detected {} devices. Currently only single-GPU training is supported!".format(torch.cuda.device_count()),
                             "Set CUDA_VISIBLE_DEVICES=<desired_gpu_id>.")
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)
        return logger, model, loader

    def get_dataset(self, args, model, data_conf, phase, n_repeat, dataset_size=-1):
        dataset_class = data_conf.dataset_spec.dataset_class
        loader = dataset_class(self.args.data_dir, data_conf, resolution=model.resolution,
                               phase=phase, shuffle=phase == "train", dataset_size=dataset_size). \
            get_data_loader(self._hp.batch_size, n_repeat)
        return loader

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    def importants_weights(self, output, inputs, k=500):
        log_weights = []
        for _ in range(k):
            z_sample = output.p.sample()  # (batch_size, latent_dim)
            x_reconstructed = self.model.decode(z_sample,
                                            cond_inputs=self.model._learned_prior_input(inputs),
                                            steps=self.model._hp.n_rollout_steps,
                                            inputs=inputs)
            mse = torch.sum((inputs.actions - x_reconstructed)**2, dim=-1)
            #p(x|z) 수정 완료
            log_px_given_z = -0.5 * torch.sum(mse, dim=-1) # 추가텀 확장
            #p(z) 애는 수정완료
            log_pz = -0.5 * torch.sum(z_sample**2, dim=-1) - 0.5 * z_sample.size(-1) * torch.log(torch.tensor(2 * math.pi))
            #q(z|x)
            log_qz_given_x = -0.5 * torch.sum((z_sample - output.p.mu)**2 / output.p.sigma**2 + torch.log(output.p.sigma**2), dim=-1)
            log_weight = log_px_given_z + log_pz - log_qz_given_x
            log_weights.append(log_weight)
        log_weights = torch.stack(log_weights, dim=0)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        log_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(k, device=z_sample.device))
        return  log_likelihood.mean().item()

def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")

def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('configs/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)


def fun_save_path(args,prefix_only = False):
    # Get the environment variable for the experiment directory
    exp_dir = os.environ['EXP_DIR']
    
    # Extract the part of the path after 'configs/'
    path = args.path.split('configs/', 1)[1]  # Extract 'skill_prior_learning/half_cheetah/FL_hierarchial_cl'
    path_components = path.split('/')[:-1]  # Extract everything except the last part (FL_hierarchial_cl)
    # Remove 'FL_hierarchial_cl' (or any last part) from the path
    if prefix_only: 
        prefix_parts = args.prefix.split('-')
        path_components.pop()
        path_components.append(prefix_parts[0])
    modified_path = '/'.join(path_components)  # Rejoin the components back
    # Extract the relevant parts from the prefix (fedadgrad and iid)
    prefix_parts = args.prefix.split('-')

    method = prefix_parts[1]  # 'fedadgrad'
    # 'iid_server' needs to be split to extract just 'iid'
    data_distribution = prefix_parts[2].split('_')[0]  # Extract 'iid' from 'iid_server'
    
    # Create the final save path
    save_path = os.path.join(exp_dir, modified_path, method, data_distribution, 'weights')
    print(save_path)
    return save_path

def gen_evaluate_fn(init_model):

    def evaluate(server_round, parameters, config):
        # 모델의 가중치를 최신 파라미터로 업데이트
        params_dict = zip(init_model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        
        # parameters update
        init_model.model.load_state_dict(state_dict,strict=False)
        
        # 데이터셋을 가져와 모델을 평가
        loss = init_model.val(server_round)
        
        # 손실과 성능 지표를 반환
        return loss, {"centralized_accuracy": 0}
    
    return evaluate

if __name__ == "__main__":
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    args = get_args()
    data_dir = args.data_dir[:-1] + "0"
    args.data_dir = data_dir
    init_model = ServerModel(args=args)
    init_path = "/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/mulstage/fedsol/hetero/weights/round-300-weights.npz"
    np_dict = np.load(init_path,allow_pickle=True)
    key_value = init_model.model.state_dict().keys()
    params_dict = zip(key_value,np_dict)
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.Tensor(np_dict[v]).to(init_model.device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    init_model.model.load_state_dict(state_dict,strict = True)
    init_model.val(100)
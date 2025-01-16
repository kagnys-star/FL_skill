import matplotlib; matplotlib.use('Agg')
import torch
import os
import time
import math

from shutil import copy
import copy as cp
import datetime
import imp
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch import autograd
from torch.optim import Adam, RMSprop, SGD
from functools import partial
from typing import Dict
from itertools import chain

from spirl.modules.variational_inference import MultivariateGaussian
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict
from spirl.components.checkpointer import get_config_path
from spirl.utils.general_utils import dummy_context, AttrDict, get_clipped_optimizer,\
                                                        AverageMeter, ParamDict
from spirl.utils.pytorch_utils import LossSpikeHook, NanGradHook, NoneGradHook, update_optimizer_lr, RAdam, SAM , ASAM , ASOL, SAM_opt, ExpSAM, \
                                        disable_running_stats , enable_running_stats, ten2ar
from spirl.modules.losses import L2Loss
from spirl.components.trainer_base import BaseTrainer
from spirl.utils.wandb import WandBLogger

import flwr as fl
from collections import OrderedDict
import code

WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'

EXCLUDE_LIST = [
    #"running",
    #"num_batches_tracked",
    #"bn",
]


class BaseClients(BaseTrainer,fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.save_dir = self.fun_save_path(prefix_only=True)
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
        self.setup_training_monitors()

        # buld dataset, model. logger, etc.
        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.logger, self.model, self.train_loader = self.build_phase(train_params, 'train')

        test_params = AttrDict(logger_class=self._hp.logger if self._hp.logger_test is None else self._hp.logger_test,
                               model_class=self._hp.model if self._hp.model_test is None else self._hp.model_test,
                               n_repeat=10,
                               dataset_size=args.val_data_size)
        self.logger_test, self.model_test, self.val_loader = self.build_phase(test_params, phase='val')
        self.model.create_beta_cycle(start=1e-3, stop=1e-1, n_epoch = 300)
        # set up optimizer + evaluator
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger_test)

        # load model params from checkpoint
        self.global_step = 0
        self.global_epoch = 0

        #save_z and log_q_zx
        #save_z and log_q_zx
        self.log_q_z_samples  = []
        self.log_q_zx_samples = []
        self.mu_samples = []


    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 16,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'minimizer ' : None,
            'lr': 1e-3,
            'gradient_clip': None,
            'init_grad_clip': 0.001,
            'init_grad_clip_step': 500,     # clip gradients in initial N steps to avoid NaNs
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'rho' : 0.1, # SAM/ ASAM optimizer
            'eta' : 0.01, # ASAM optimizer
            'mu' : 0., #Proximal term
            'adam_beta': 0.9,       # beta1 param in Adam
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
            'logging_target': 'wandb',
        })
        return default_dict


    def train(self) :
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        for epoch in range(self._hp.num_epochs):
            self.train_epoch(epoch)
            self.global_epoch += 1
            self.aux_log()


    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            self.optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses.total.value.backward()
            self.aux_info(output)
            self.call_hooks(inputs, output, losses, epoch)
            if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            self.optimizer.step()
            self.model.step()

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()

            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            if self.log_outputs_now:
                self.log_outputs( output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)
            del output, losses
            self.global_step = self.global_step + 1

    def val(self):
        print('Running Testing')
        if self.args.test_prediction:
            start = time.time()
            self.model_test.load_state_dict(self.model.state_dict())
            losses_meter = RecursiveAverageMeter()
            iw_meter = 0
            self.model_test.eval()
            self.evaluator.reset()
            with autograd.no_grad():
                for sample_batched in self.val_loader:
                    
                    inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

                    # run evaluator with val-mode model
                    with self.model_test.val_mode():
                        self.evaluator.eval(inputs, self.model_test)

                    # run non-val-mode model (inference) to check overfitting
                    output = self.model_test(inputs)
                    losses = self.model_test.loss(output, inputs)
                    output.q_reconstruction = self.model.decode(output.z_p,
                                            cond_inputs=self.model._learned_prior_input(inputs),
                                            steps=self.model._hp.n_rollout_steps,
                                            inputs=inputs)
                    losses.prior_mse = L2Loss(1.0)(inputs.actions,(output.q_reconstruction))
                    iw_meter += self.importants_weights(output, inputs)
                    losses_meter.update(losses)
                    del losses
                
                if not self.args.dont_save:
                    if self.evaluator is not None:
                        self.evaluator.dump_results(self.global_step)
                    self.model_test.log_outputs(output, inputs, losses_meter.avg, self.global_step,
                                                log_images=False, phase='val', **self._logging_kwargs)
                    self.logger.log_scalar(iw_meter/(len(self.val_loader)), "LL",self.global_step, phase='val')
                    print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                           .format(losses_meter.avg.total.value.item(), time.time() - start)))
            del output
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

    def setup_training_monitors(self):
        self.training_context = autograd.detect_anomaly if self.args.detect_anomaly else dummy_context
        self.hooks = []
        self.hooks.append(LossSpikeHook('sg_img_mse_train'))
        self.hooks.append(NanGradHook(self))
        self.hooks.append(NoneGradHook(self))

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

    def get_optimizer_class(self):
        optim = self._hp.optimizer
        if optim == 'adam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=Adam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'radam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RAdam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'rmsprop':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RMSprop, momentum=self._hp.momentum)
        elif optim == 'sgd':
            get_optim = partial(get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum)
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optim))
        return partial(get_optim, gradient_clip=self._hp.gradient_clip)

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    @property
    def log_images_now(self):
        return self.global_step % self.log_images_interval == 0

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0
    
    def aux_info(self,outputs):
        z_prior = MultivariateGaussian(torch.zeros_like(outputs.q.mu), torch.zeros_like(outputs.q.sigma))
        self.log_q_zx_samples.append(ten2ar(outputs.q.log_prob(outputs.z)))
        self.log_q_z_samples.append(ten2ar(z_prior.log_prob(outputs.z)))
        self.mu_samples.append(ten2ar(outputs.q.mu))

    def log_outputs(self, output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len):
        self.model.log_outputs(output, inputs, losses, self.global_step,
                                log_images=False, phase='train', **self._logging_kwargs)
        with autograd.no_grad():
            output.q_reconstruction = self.model.decode(output.z_p,
                                    cond_inputs=self.model._learned_prior_input(inputs),
                                    steps=self.model._hp.n_rollout_steps,
                                    inputs=inputs)
            self.logger.log_scalar(torch.nn.MSELoss(reduction='none')(inputs.actions,(output.q_reconstruction )), "prior_mse",self.global_step, phase='train')
        print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none',
                                    self._hp.exp_path))
        print(('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.global_step, epoch, self.batch_idx, len(self.train_loader),
                100. * self.batch_idx / len(self.train_loader), losses.total.value.item())))

        print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                .format(data_load_time.avg,
                        batch_time.avg - upto_log_time.avg,
                        upto_log_time.avg - data_load_time.avg,
                        batch_time.avg))
        togo_train_time = batch_time.avg * (self._hp.num_epochs - epoch) * epoch_len / 3600.
        print('ETA: {:.2f}h'.format(togo_train_time))

    def aux_log(self):
        log_q_zx = np.concatenate(np.array(self.log_q_zx_samples), axis=0)
        log_q_z = np.concatenate(np.array(self.log_q_z_samples), axis=0)
        mi_estimate = (log_q_zx - log_q_z)
        mu_s = np.concatenate(np.array(self.mu_samples), axis=0)
        z_var = np.var(mu_s, axis=0, ddof=1)
        threshold = 0.01
        self.log_q_z_samples = []
        self.log_q_zx_samples= []
        self.mu_samples= []
        activated_units = (z_var > threshold).astype(float)
        self.logger.log_scalar(np.mean(activated_units), "AU", self.global_step, "train")
        self.logger.log_scalar(np.mean(mi_estimate), "mi_estimate", self.global_step, "train")

    def importants_weights(self, output, inputs , k=5):
        z_prior = MultivariateGaussian(torch.zeros_like(output.q.mu),torch.zeros_like(output.q.sigma))
        log_weights = []
        for _ in range(k):
            z_sample = output.q.sample()  # (batch_size, latent_dim)
            x_reconstructed = self.model.decode(z_sample,
                                            cond_inputs=self.model._learned_prior_input(inputs),
                                            steps=self.model._hp.n_rollout_steps,
                                            inputs=inputs)
            mse = torch.sum((inputs.actions - x_reconstructed)**2, dim=-1)
            log_px_given_z = z_prior.log_prob(mse) # 추가텀 확장
            log_pz = z_prior.log_prob(z_sample)
            log_qz_given_x = output.q.log_prob(z_sample)
            log_weight = log_px_given_z + log_pz - log_qz_given_x
            log_weights.append(log_weight)
        log_weights = torch.stack(log_weights, dim=0)
        log_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(k, device=z_sample.device))
        return log_likelihood.mean().item()
        

    def fun_save_path(self,prefix_only = False):
        # Get the environment variable for the experiment directory
        exp_dir = os.environ['EXP_DIR']
        
        # Extract the part of the path after 'configs/'
        path = self.args.path.split('configs/', 1)[1]  # Extract 'skill_prior_learning/half_cheetah/FL_hierarchial_cl'
        path_components = path.split('/')[:-1]  # Extract everything except the last part (FL_hierarchial_cl)
        # Remove 'FL_hierarchial_cl' (or any last part) from the path
        if prefix_only: 
            prefix_parts = self.args.prefix.split('-')
            path_components.pop()
            path_components.append(prefix_parts[0])
        modified_path = '/'.join(path_components)  # Rejoin the components back
        # Extract the relevant parts from the prefix (fedadgrad and iid)
        prefix_parts = self.args.prefix.split('-')

        method = prefix_parts[1]  # 'fedadgrad'
        # 'iid_server' needs to be split to extract just 'iid'
        data_distribution = prefix_parts[2].split('_')[0]  # Extract 'iid' from 'iid_server'
        
        client_num = self.args.data_dir.split('/')[-1]
        # Create the final save path
        save_path = os.path.join(exp_dir, modified_path, method, data_distribution, client_num, 'weights')
        os.makedirs(save_path , exist_ok=True)
        return save_path
    ########################
    ### flower functions ### 
    ########################

    def get_parameters(self,config):
        if (self.global_epoch % 10 == 0):
            aggregated_ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            np.savez(os.path.join(self.save_dir,f"round-{self.global_epoch}-weights.npz"), *aggregated_ndarrays)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
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
        self.model.load_state_dict(state_dict,strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if "learning_rate" in config:
            self._hp.lr = config["learning_rate"]
        self.train()
        return self.get_parameters(config), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.val()
        return float(loss), len(self.val_loader), {"accuracy": 0}

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


class ASAMClients(BaseClients):
    def __init__(self, args):
        super().__init__(args)
    
    def train(self) :
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        if self._hp.minimizer == "sam":
            self.minimizer = SAM_opt(base_optimizer= self.optimizer, params = self.model.parameters(), rho=self._hp.rho)
        else:
            self.minimizer = SAM_opt(base_optimizer= self.optimizer, params = self.model.parameters(), rho=self._hp.rho, adaptive=True)
        for epoch in range(self._hp.num_epochs):
            self.train_epoch(epoch)
            self.global_epoch += 1
            self.aux_log()
    
    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            enable_running_stats(self.model)
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses.total.value.backward()
            self.call_hooks(inputs, output, losses, epoch)

            self.minimizer.ascent_step()  # Perform the ascent step
            # Recompute forward pass after the ascent step
            disable_running_stats(self.model)
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            #if self.global_step < self._hp.init_grad_clip_step:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            #else:
            #    parameters_to_clip = [param for name, param in self.model.named_parameters() if 'p.0' in name]
            #    torch.nn.utils.clip_grad_norm_(parameters_to_clip, self._hp.init_grad_clip)
            losses.total.value.backward()  # Second backward pass
            self.minimizer.descent_step()
            self.model.step()
            self.aux_info(output)

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()

            if self.log_outputs_now:
                self.log_outputs(output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)

            del output, losses
            self.global_step = self.global_step + 1

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if "learning_rate" in config:
            self._hp.lr = config["learning_rate"]
        self.train()
        return self.get_parameters(config), len(self.train_loader), {}


class FP_Clients(BaseClients):

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.global_weights = cp.deepcopy(self.model.state_dict())


    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            self.optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses = self.add_loss_drift(losses)
            losses.total.value.backward()
            self.call_hooks(inputs, output, losses, epoch)

            if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)

            self.optimizer.step()
            self.model.step()
            self.aux_info(output)

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.log_outputs_now:
                self.log_outputs(output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)

            del output, losses
            self.global_step = self.global_step + 1


    def add_loss_drift(self,losses):
        if self._hp.mu == 0:
            return losses
        local_weights = self.model.state_dict()
        exception_list = ["num_batches_tracked", "running_mean", "running_var" , "log_sigma"]
        prox_term = 0.0
        for key ,params in local_weights.items():
            if any(excep in key for excep in exception_list):  # Improved condition check
                continue
            # Collect tensors in lists
            prox_term += torch.norm((params - self.global_weights[key])) ** 2
        losses.proximal = AttrDict(value=prox_term, weight=(self._hp.mu/2))
        if hasattr(losses, 'total'):
            del losses.total
        losses.total = self.model._compute_total_loss(losses)
        return losses


class FN_Clients(FP_Clients):

    def train(self) :
        self.tau = 0
        super().train()

    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            self.optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses = self.add_loss_drift(losses)
            losses.total.value.backward()
            self.call_hooks(inputs, output, losses, epoch)

            if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            else:
                parameters_to_clip = [param for name, param in self.model.named_parameters() if 'p.0' in name]
                torch.nn.utils.clip_grad_norm_(parameters_to_clip, self._hp.init_grad_clip)
            self.optimizer.step()
            self.model.step()
            self.aux_info(output)

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.log_outputs_now:
                self.log_outputs(output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)

            del output, losses
            self.tau += 1
            self.global_step = self.global_step + 1


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if "learning_rate" in config:
            self._hp.lr = config["learning_rate"]
        self.train()
        grad_scaling_factor = self.get_gradient_scaling()
        return self.get_parameters(config), len(self.train_loader), grad_scaling_factor


    def get_parameters(self, config):
        if (self.global_epoch % 10 == 0):
            aggregated_ndarrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            np.savez(os.path.join(self.save_dir,f"round-{self.global_epoch}-weights.npz"), *aggregated_ndarrays)
        return [(self.global_weights[key].cpu().numpy() - param.cpu().numpy()) for key, param in self.model.state_dict().items()]
    
    def get_gradient_scaling(self):
        grad_scaling_factor = {}
        #etamu = self._hp.lr * self._hp.mu
        #coeff = (self.tau - self._hp.momentum * (1 - pow(self._hp.momentum, self.tau)) / (1 - self._hp.momentum)) / (1 - self._hp.momentum)
        #if etamu != 0:
        #    coeff *= (1 - etamu)
        grad_scaling_factor['tau'] = self.tau
        return grad_scaling_factor


class SF_Clients(BaseClients):

    def __init__(self, args):
        super().__init__(args)
        self.control_var = {key: np.zeros(val.cpu().numpy().shape) for key, val in self.model.state_dict().items()}
        # 클라이언트 제어 변수 c_i
        start = args.prefix.find("client_") + len("client_")  # 'client_' 바로 뒤에서 시작
        end = args.prefix.rfind("-")  # 마지막 '-' 앞에서 끝
        self.client_num = args.prefix[start:end]
        initial_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        self.weight_shapes = list(map(lambda arr: arr.shape, initial_parameters))


    def train(self) :
        self.cnt = 0
        for epoch in range(self._hp.num_epochs):
            self.train_epoch(epoch)
            self.global_epoch += 1
            self.aux_log()


    def train_epoch(self, epoch):
        self.model.train()
        #self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)

        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            self.optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses.total.value.backward()
            self.call_hooks(inputs, output, losses, epoch)

            ####client_drifit####
            exception_list = ["log_sigma" , "num_batches_tracked", "running_mean" , "running_var" ]
            for key , param in self.model.named_parameters():
                if any(excep in key for excep in exception_list):  # Improved condition check
                    continue
                if param.grad is not None:  # 그라디언트가 존재할 경우
                    delta_control_variates = torch.tensor(self.delta_control_variates[key], device=self.device, dtype=param.grad.dtype)
                
                    # 크기 출력
                    #print(key)
                    #print(f"control_var_tensor shape: {control_var_tensor.shape}, server_control_var_tensor shape: {server_control_var_tensor.shape}, param.grad shape: {param.grad.shape}")
                    
                    # 보정 연산 수행
                    param.grad.data += delta_control_variates


            #if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            parameters_to_clip = [param for name, param in self.model.named_parameters() if 'p.0' in name]
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, self._hp.init_grad_clip)
            self.optimizer.step()
            self.model.step()
   
            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            if self.log_outputs_now:
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       log_images=False, phase='train', **self._logging_kwargs)
            batch_time.update(time.time() - end)
            end = time.time()

            if self.log_outputs_now:
                print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none',
                                          self._hp.exp_path))
                print(('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.global_step, epoch, self.batch_idx, len(self.train_loader),
                        100. * self.batch_idx / len(self.train_loader), losses.total.value.item())))

                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))
                togo_train_time = batch_time.avg * (self._hp.num_epochs - epoch) * epoch_len / 3600.
                print('ETA: {:.2f}h'.format(togo_train_time))

            del output, losses
            self.global_step = self.global_step + 1
            self.cnt += 1


    def set_parameters(self, parameters):
        server_weights = parameters[ : len(self.weight_shapes)]
        server_covariates = parameters[len(self.weight_shapes) : ]
        super().set_parameters(server_weights)
        self.global_weights = cp.deepcopy(self.model.state_dict())
        #drift_cal_fist and use it c - c_i
        self.delta_control_variates = cp.deepcopy(self.control_var)
        for i, key in enumerate(self.delta_control_variates.keys()):
                self.delta_control_variates[key] = server_covariates[i] - self.control_var[key]


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        weights = self.update_control_variate()
        return weights, len(self.train_loader), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.val()
        return float(loss), len(self.val_loader), {"accuracy": 0}


    def update_control_variate(self):
        state_dict = self.model.state_dict()
        control_var_plus = cp.deepcopy(self.global_weights)
        updated_client_control_variates = cp.deepcopy(self.control_var)
        exception_list = ["log_sigma", "num_batches_tracked", "running_mean" , "running_var"]
        scaling_coefficient = 1 / (self.cnt * self._hp.lr)
        self.logger.log_scalar(scaling_coefficient , "scaling_coefficient", self.global_step, phase = 'train')
        for key in state_dict.keys():
            if any(excep in key for excep in exception_list): 
                control_var_plus[key] = np.zeros(self.control_var[key].shape)
                updated_client_control_variates[key] = np.zeros(self.control_var[key].shape)
            else:
                control_var_plus[key] = scaling_coefficient * (self.global_weights[key].cpu().numpy() - state_dict[key].cpu().numpy()) - self.delta_control_variates[key]
                updated_client_control_variates[key] = control_var_plus[key] - self.control_var[key]
        self.control_var = control_var_plus
        #file_name = f"c_i_client_{self.client_num}-{self.global_epoch}.pickle"
        #with open(file_name,'wb') as fw:
        #    pickle.dump(self.control_var, fw)

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()] + [val for _, val in updated_client_control_variates.items()]


class FS_Clients(BaseClients):

    def __init__(self, args):
        super().__init__(args)
        self.global_weights = cp.deepcopy(self.model.state_dict())

    def train(self) :
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        if self._hp.minimizer == "ASOL":
            self.minimizer = ExpSAM(base_optimizer=self.optimizer, params = self.model.parameters(), rho=self._hp.rho)
        elif self._hp.minimizer == "SAM":
            self.minimizer = SAM_opt(base_optimizer=self.optimizer, params = self.model.parameters(), rho=self._hp.rho)
        for epoch in range(self._hp.num_epochs):
            self.train_epoch(epoch)
            self.global_epoch += 1
            self.aux_log()


    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            enable_running_stats(self.model)
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses = self.add_loss_drift(losses)
            losses.total.value.backward()
            #proximal = self.add_loss_drift()
            #proximal.total.value.backward()
            self.minimizer.ascent_step()  # Perform the ascent step
            # Recompute forward pass after the ascent step
            disable_running_stats(self.model)
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            losses.total.value.backward()  # Second backward pass
            self.call_hooks(inputs, output, losses, epoch)
            self.minimizer.descent_step()
            self.model.step()
            self.aux_info(output)

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.log_outputs_now:
                self.log_outputs(output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)

            del output, losses
            self.global_step = self.global_step + 1


    def add_loss_drift(self, losses):
        new_losses = AttrDict()
        exception_list = [ "log_sigma", "num_batches_tracked", "running_mean", "running_var" ]
        prox_term = 0.0
        for key ,params in self.model.named_parameters():
            if any(excep in key for excep in exception_list):  # Improved condition check
                continue
            # Collect tensors in lists
            prox_term += torch.norm((params - self.global_weights[key])) ** 2
        if hasattr(losses, 'total'):
            del losses.total
        new_losses.proximal = AttrDict(value=prox_term, weight=self._hp.mu/2)
        new_losses.total = AttrDict(value=(new_losses.proximal.value * new_losses.proximal.weight))
        return new_losses

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.global_weights = cp.deepcopy(self.model.state_dict())


class FD_Clients(FP_Clients):

    def __init__(self, args):
        super().__init__(args)
        self.local_grads = cp.deepcopy(self.model.state_dict())
        for key in self.local_grads:
            self.local_grads[key] = self.local_grads[key].float()
            self.local_grads[key].zero_()  
        #self.global_weights = cp.deepcopy(self.model_test.state_dict())
        #self.caculate_grad()
        #self.train()


    def add_loss_drift(self,losses):
        exception_list = ["log_sigma", "num_batches_tracked", "running_mean", "running_var"]
        lin_penalty = 0.0
        sq_penalty = 0.0
        for key ,params in self.model.state_dict().items():
            if any(excep in key for excep in exception_list):  # Improved condition check
                continue
            # gradients_caculator
            lin_penalty += torch.sum(torch.mul(self.local_grads[key], params))
            #proximal_term
            sq_penalty += torch.sum(torch.norm(self.global_weights[key] - params, 2))
        losses.lin_penalty  = AttrDict(value=lin_penalty, weight=-1)
        losses.sq_penalty = AttrDict(value=sq_penalty, weight=(self._hp.mu/2))
        if hasattr(losses, 'total'):
            del losses.total
        losses.total = self.model._compute_total_loss(losses)
        return losses


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if "learning_rate" in config:
            self._hp.lr = config["learning_rate"]
        self.train()
        self.caculate_grad()
        return self.get_parameters(config), len(self.train_loader), {}


#식 2번째
    def caculate_grad(self):
        for key, param in self.model.state_dict().items() :
            self.local_grads[key] -= self._hp.mu *(self.global_weights[key].float() - param.float())


class Test_Clients(BaseClients):
    def __init__(self, args):
        from spirl.models.spt_mdl import MMDEncoder
        super().__init__(args)
        spt_config = AttrDict(
            state_dim=self.conf.model.state_dim,
            action_dim=self.conf.model.n_actions,
            n_rollout_steps=self.conf.model.n_rollout_steps,
            nz_enc=self.conf.model.nz_enc,
            nz_mid=self.conf.model.nz_mid,
            n_processing_layers=self.conf.model.n_processing_layers,
            cond_decode=self.conf.model.cond_decode,
            device =self.conf.model.device,
            batch_size = self.conf.model.batch_size,
            kernel_type = "IMQ"
        )
        self.sp_encoder = MMDEncoder(spt_config)
        self.sp_encoder.load_weights('global',self.model.q.state_dict())



    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            self.optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses = self.sp_encoder.calculate(inputs, output,losses)
            losses.total.value.backward()
            self.aux_info(output)
            self.call_hooks(inputs, output, losses, epoch)
            if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            self.optimizer.step()
            self.model.step()

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()

            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            if self.log_outputs_now:
                self.log_outputs( output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)
            del output, losses
            self.global_step = self.global_step + 1

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.sp_encoder.load_weights('global',self.model.q.state_dict())


class Mulopt_Clients(BaseClients):
    def __init__(self, args):
        super().__init__(args)
        del self.optimizer
        self.model.dataset_size = len(self.train_loader)
        self.E_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, chain(self.model.decoder.parameters(),self.model.q.parameters())), lr=self._hp.lr)
        self.P_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.p.parameters()), lr=self._hp.lr)
        #self.train_epoch(0)

    
    def train(self) :
        self.E_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, chain(self.model.decoder.parameters(),self.model.q.parameters())), lr=self._hp.lr)
        self.P_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.p.parameters()), lr=self._hp.lr)
        for epoch in range(self._hp.num_epochs):
            self.train_epoch(epoch)
            self.global_epoch += 1
            self.aux_log()
    
    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            self.E_optimizer.zero_grad()
            self.P_optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)

            
            losses.total.value.backward()
            #if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
            torch.nn.utils.clip_grad_norm_(chain(self.model.decoder.parameters(),self.model.q.parameters()), self._hp.init_grad_clip)
            self.E_optimizer.step()

            losses2 = losses.q_hat_loss.value * losses.q_hat_loss.weight
            losses2.backward()
            #if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
            torch.nn.utils.clip_grad_norm_(self.model.p.parameters(), self._hp.init_grad_clip)
            self.P_optimizer.step()
            self.aux_info(output)
            self.call_hooks(inputs, output, losses, epoch)
   
            self.model.step()

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()

            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            if self.log_outputs_now:
                self.log_outputs( output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)
            del output, losses
            self.global_step = self.global_step + 1


class VAEGAN_Clients(BaseClients):
    def __init__(self, args):
        super().__init__(args)
        del self.optimizer
        self.E_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.q.parameters()), lr=self._hp.lr)
        self.P_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.p.parameters()), lr=self._hp.lr)
        self.D_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.decoder.parameters()), lr=self._hp.lr)
        self.Dis_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.discriminator.parameters()), lr=self._hp.lr)
    
    def train(self) :
        self.E_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.q.parameters()), lr=self._hp.lr)
        self.P_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.p.parameters()), lr=self._hp.lr)
        self.D_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.decoder.parameters()), lr=self._hp.lr)
        self.Dis_optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.discriminator.parameters()), lr=self._hp.lr)
        for epoch in range(self._hp.num_epochs):
            self.train_epoch(epoch)
            self.global_epoch += 1
            self.aux_log()
    
    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            Dis_loss = losses.gan_loss.value *losses.gan_loss.weight
            Dec_loss =  losses.rec_mse.value *losses.rec_mse.weight -(losses.gan_loss.value *losses.gan_loss.weight)
            Enc_loss =  losses.kl_loss.value *losses.kl_loss.weight + losses.gan_loss.value *losses.gan_loss.weight
            losses2 = self.model.loss_q(output)
            if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            self.Dis_optimizer.zero_grad()
            Dis_loss.backward(retain_graph=True)
            self.Dis_optimizer.step()
            self.D_optimizer.zero_grad()
            Dec_loss.backward(retain_graph=True)
            self.D_optimizer.step()
            self.E_optimizer.zero_grad()
            Enc_loss.backward()
            self.E_optimizer.step()
            self.P_optimizer.zero_grad()
            losses2.total.value.backward()
            self.P_optimizer.step()
            self.aux_info(output)
            self.call_hooks(inputs, output, losses, epoch)
            self.model.step()

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()

            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            if self.log_outputs_now:
                self.log_outputs( output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)
            del output, losses
            self.global_step = self.global_step + 1

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d  or "discriminator" in d:
                l.append(d)
        for d in l :
            del( state_dict[d] )
        # parameters update
        self.model.load_state_dict(state_dict,strict=False)



class Dualenc_Clients(BaseClients):

    def train_epoch(self, epoch):
        self.model.train()
        self.model.update_cycle_beta(self.global_epoch)
        self.model.round = self.global_epoch
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        print('starting epoch ', epoch)
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            self.optimizer.zero_grad()
            losses.total.value.backward()
            if self.global_step < self._hp.init_grad_clip_step:
            # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            self.optimizer.step()

            self.aux_info(output)
            self.call_hooks(inputs, output, losses, epoch)
   
            self.model.step()

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()

            upto_log_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            if self.log_outputs_now:
                self.log_outputs( output, inputs,epoch,losses,data_load_time,batch_time,upto_log_time,epoch_len)
            del output, losses
            self.global_step = self.global_step + 1


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d  or "q" in d:
                l.append(d)
        for d in l :
            del( state_dict[d] )
        # parameters update
        self.model.load_state_dict(state_dict,strict=False)
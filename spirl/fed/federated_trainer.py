import matplotlib; matplotlib.use('Agg')
import torch
import os
import imp
import numpy as np
import time
import random

import pprint
from torch import autograd
from torch.optim import Adam, RMSprop, SGD
from functools import partial
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict
from spirl.components.checkpointer import get_config_path
from spirl.utils.general_utils import  AttrDict, get_clipped_optimizer, ParamDict, AverageMeter
from spirl.utils.pytorch_utils import  RAdam
from spirl.components.trainer_base import BaseTrainer
from spirl.utils.general_utils import prefix_dict
import logging

logging.basicConfig(level=logging.INFO)

class ModelTrainer(BaseTrainer):
    def __init__(self, config ,cid ,logger):
        self.setup_device()
        self.logger = logger
        set_seeds()
        # set up params
        self.conf = conf = self.get_config(config.path)
        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)
        self._hp.overwrite(config.params)  # override defaults with config file
        self.conf = self.postprocess_conf(conf)
        pprint.pprint(self._hp)
        self.prefix = config.prefix
        self.log_outputs_interval = config.val_interval
        self._hp.data_dir =os.path.join(config.init_data_dir, 'FL_{}'.format(cid))
        self.log_dir = os.path.join(self._hp.exp_path, 'events')
        self.model, self.train_loader, self.val_loader = self.build_phase(logger)
        # set up optimizer + evaluator
        self._logging_kwargs = AttrDict()
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger)
        # load model params from checkpoint
        self.global_step, self.start_epoch = 0, 0
    
    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 128,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'lr': 1e-3,
            'gradient_clip': None,
            'init_grad_clip': 0.001,
            'init_grad_clip_step': 100,     # clip gradients in initial N steps to avoid NaNs
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
        })
        return default_dict
    
    def train(self, params):
        for epoch in range(0 , self._hp.num_epochs):
            self.train_epoch(epoch)
        self.start_epoch = 0
        return self.global_step

    def train_epoch(self, epoch):
        self.model.train()
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        
        print('starting epoch ', epoch)
        log_len = len(self.train_loader) - 1
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            self.optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses.total.value.backward()

            if self.global_step < self._hp.init_grad_clip_step:
                # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
            self.optimizer.step()
            self.model.step()
            
            upto_log_time.update(time.time() - end)
            #if self.log_outputs_now and not self.args.dont_save:
            if log_len == self.batch_idx :
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       log_images=False, phase=f'train/{self.prefix}', **self._logging_kwargs)
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

    def val(self):
        print('Running Testing')
        losses_meter = RecursiveAverageMeter()
        self.model.eval()
        self.evaluator.reset()
        with autograd.no_grad():
            for sample_batched in self.val_loader:
                
                inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

                # run evaluator with val-mode model
                with self.model.val_mode():
                    self.evaluator.eval(inputs, self.model)

                # run non-val-mode model (inference) to check overfitting
                output = self.model(inputs)
                losses = self.model.loss(output, inputs)

                losses_meter.update(losses)
                del losses
                
            if self.evaluator is not None:
                self.evaluator.dump_results(self.global_step)
            self.model.log_outputs(output, inputs, losses_meter.avg, self.global_step,
                                        log_images=False, phase=f'val/{self.prefix}', **self._logging_kwargs)
        return losses_meter.avg.total.value.item()


    def setup_device(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    def get_config(self, path):
        conf = AttrDict()
        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(path)

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
        return conf

    def postprocess_conf(self, conf):
        conf.model['batch_size'] = self._hp.batch_size if not torch.cuda.is_available() \
            else int(self._hp.batch_size / torch.cuda.device_count())
        conf.model.update(conf.data.dataset_spec)
        conf.model['device'] = conf.data['device'] = self.device.type
        return conf

    def build_phase(self,logger):
        model = self._hp.model(self.conf.model, logger)
        if torch.cuda.device_count() > 1:
            raise ValueError("Detected {} devices. Currently only single-GPU training is supported!".format(torch.cuda.device_count()),
                             "Set CUDA_VISIBLE_DEVICES=<desired_gpu_id>.")
        model = model.to(self.device)
        model.device = self.device
        train_loader = self.get_dataset(model.resolution, self.conf.data, "train", self._hp.epoch_cycles_train, -1)
        val_loader = self.get_dataset(model.resolution, self.conf.data, 'val', 1, self._hp.batch_size)
        return model, train_loader , val_loader

    def get_dataset(self, resolution, data_conf, phase, n_repeat, dataset_size=-1):
        dataset_class = data_conf.dataset_spec.dataset_class
        loader = dataset_class(self._hp.data_dir, data_conf, resolution=resolution,
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
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0

def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

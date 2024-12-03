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
from torch.optim import Adam, RMSprop, SGD
from functools import partial

from spirl.components.data_loader import RandomVideoDataset
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict
from spirl.components.checkpointer import get_config_path
from spirl.utils.general_utils import  AttrDict, get_clipped_optimizer, \
                                                         ParamDict
from spirl.utils.pytorch_utils import  RAdam, ProxSGD, SAM
from spirl.components.trainer_base import BaseTrainer
from spirl.utils.wandb import WandBLogger
from collections import OrderedDict

WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'


class Eval_model(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.setup_device()
        # set up params
        self.conf = conf = self.get_config()
        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        #self.data_dir =os.path.join(data_dir, 'FL_{}'.format(cid))
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
        self.conf = self.postprocess_conf(conf)
        if args.deterministic: set_seeds()
        # set up logging + training monitoring
        self.writer = self.setup_logging(conf, self.log_dir)
        # buld dataset, model. logger, etc.
        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.logger, self.model, self.train_loader = self.build_phase(train_params, 'train')

        # set up optimizer + evaluator
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger_test)


    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 32,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'lr': 1e-3,
            'gradient_clip': None,
            'init_grad_clip': 0.001,
            'init_grad_clip_step': 500,     # clip gradients in initial N steps to avoid NaNs
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
            'logging_target': 'wandb',
        })
        return default_dict


    def val(self, params):
        print('Running Testing')
        if self.args.test_prediction:
            start = time.time()
            self.model.load_state_dict(params)
            losses_meter = RecursiveAverageMeter()
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
                    #self.logger_test.log_scalar(score, "Score", self.global_step, "eval/Score")
                    losses = self.model.loss(output, inputs)

                    losses_meter.update(losses)
                    del losses
                
                if not self.args.dont_save:
                    if self.evaluator is not None:
                        self.evaluator.dump_results(self.global_step)

                    self.model.log_outputs(output, inputs, losses_meter.avg, self.global_step,
                                                log_images=False, phase='val', **self._logging_kwargs)
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
            #print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
            #model = DataParallelWrapper(model)
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)
        return logger, model, loader

    def get_dataset(self, args, model, data_conf, phase, n_repeat, dataset_size=-1):
        if args.feed_random_data:
            dataset_class = RandomVideoDataset
        else:
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
        elif optim == 'proxsgd':  # New optimizer added here
            return partial(ProxSGD, gmf=self._hp.gmf, mu=self._hp.mu, momentum=self._hp.momentum,ratio=self._hp.ratio)
        elif optim == 'asam':  # New optimizer added here
            return  partial(SAM,base_optimizer=SGD ,adaptive=True, weight_decay = 4e-4, momentum = 0.9)
        elif optim == 'sam':  # New optimizer added here
            return partial(SAM,base_optimizer=SGD, adaptive=False , weight_decay = 4e-4 , momentum = 0.9)
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


import matplotlib; matplotlib.use('Agg')
import torch
import os
import time
from shutil import copy
import datetime
import imp
from tensorboardX import SummaryWriter
import numpy as np
import math
import random
from torch import autograd
from torch.optim import Adam, RMSprop, SGD
from functools import partial
from spirl.modules.losses import L2Loss
from spirl.components.data_loader import RandomVideoDataset
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict ,pretty_print
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.utils.general_utils import dummy_context, AttrDict, get_clipped_optimizer, \
                                                        AverageMeter, ParamDict
from spirl.utils.pytorch_utils import LossSpikeHook, NanGradHook, NoneGradHook, \
                                                        DataParallelWrapper, RAdam , SAM , ASAM , ten2ar
from spirl.components.trainer_base import BaseTrainer
from spirl.utils.wandb import WandBLogger
from spirl.components.params import get_args
import random

random_seed = 8888
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'


class ModelTrainer(BaseTrainer):
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
        self.setup_training_monitors()
        
        # buld dataset, model. logger, etc.
        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.logger, self.model, self.train_loader = self.build_phase(train_params, 'train')

        test_params = AttrDict(logger_class=self._hp.logger if self._hp.logger_test is None else self._hp.logger_test,
                               model_class=self._hp.model if self._hp.model_test is None else self._hp.model_test,
                               n_repeat=1,
                               dataset_size=args.val_data_size)
        self.logger_test, self.model_test, self.val_loader = self.build_phase(test_params, phase='val')

        # set up optimizer + evaluator
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger_test)

        # load model params from checkpoint
        self.global_step, start_epoch = 0, 0
        self.z_samples  = []
        self.log_q_zx_samples = []
        if args.resume or conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, conf.ckpt_path)
        exit()
        if args.val_sweep:
            self.run_val_sweep()
        elif args.train:
            self.train(start_epoch)
        else:
            self.val()

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
            'num_epochs': 100,
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
            'logging_target': 'wandb',
        })
        return default_dict
    
    def train(self, start_epoch):
        if not self.args.skip_first_val:
            self.val()
            
        for epoch in range(start_epoch, self._hp.num_epochs):
            self.train_epoch(epoch)
            self.aux_log()
        
            if not self.args.dont_save:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.model.state_dict(),
                    #'optimizer': self.optimizer.state_dict(),
                },  os.path.join(self._hp.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))

            if epoch % self.args.val_interval == 0:
                self.val()

    def train_epoch(self, epoch):
        self.model.train()
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
            with self.training_context():
                if isinstance(self.optimizer, ASAM):
                    output = self.model(inputs)
                    losses = self.model.loss(output, inputs)
                    losses.total.value.backward()
                    self.call_hooks(inputs, output, losses, epoch)
                    self.optimizer.ascent_step()  # Perform the ascent step
                    # Recompute forward pass after the ascent step
                    output = self.model(inputs)
                    losses = self.model.loss(output, inputs)
                    if self.global_step < self._hp.init_grad_clip_step:
                    # clip gradients in initial steps to avoid NaN gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
                    losses.total.value.backward()  # Second backward pass
                    self.optimizer.descent_step()
                else:
                    self.optimizer.zero_grad()
                    output = self.model(inputs)

                    losses = self.model.loss(output, inputs)
                    losses.total.value.backward()
                    self.call_hooks(inputs, output, losses, epoch)
                    if self.global_step < self._hp.init_grad_clip_step:
                    # clip gradients in initial steps to avoid NaN gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hp.init_grad_clip)
                    self.optimizer.step()
                self.aux_info(output)
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
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(conf.conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))
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

        loader = dataset_class(self._hp.data_dir, data_conf, resolution=model.resolution,
                               phase=phase, shuffle=phase == "train", dataset_size=dataset_size). \
            get_data_loader(self._hp.batch_size, n_repeat)
        return loader

    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.model,
                                           load_step=True, load_opt=True, optimizer=self.optimizer,
                                           strict=self.args.strict_weight_loading)
        self.model.to(self.model.device)
        return start_epoch

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

    def run_val_sweep(self):
        epochs = CheckpointHandler.get_epochs(os.path.join(self._hp.exp_path, 'weights'))
        for epoch in list(sorted(epochs))[::2]:
            self.resume(epoch)
            self.val()
        return

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    @property
    def log_images_now(self):
        return self.global_step % self.log_images_interval == 0

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0
    
    def aux_info(self,outputs):
        self.log_q_zx_samples.append(ten2ar(outputs.q.log_prob(outputs.z)))
        self.z_samples.append(ten2ar(outputs.z))

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
        z_s = np.concatenate(np.array(self.z_samples), axis=0)
        log_q_zx = np.concatenate(np.array(self.log_q_zx_samples), axis=0)
        z_mean = np.mean(z_s,axis=0)
        z_var = np.var(z_s,axis=0)
        log_q_z = -0.5 * np.sum(((z_s - z_mean) ** 2 / z_var + np.log(z_var) + np.log(2 * np.pi)),axis=1)
        mi_estimate = (log_q_zx - log_q_z)
        threshold = 0.01
        self.z_samples = []
        self.log_q_zx_samples= []

        activated_units = (z_var > threshold).astype(float)
        self.logger.log_scalar(np.mean(activated_units), "AU", self.global_step, "train")
        self.logger.log_scalar(np.mean(mi_estimate), "mi_estimate", self.global_step, "train")

    def importants_weights(self, output, inputs , k=5):
        sigma = 1.0
        log_weights = []
        for _ in range(k):
            z_sample = output.p.sample()  # (batch_size, latent_dim)
            x_reconstructed = self.model.decode(z_sample,
                                            cond_inputs=self.model._learned_prior_input(inputs),
                                            steps=self.model._hp.n_rollout_steps,
                                            inputs=inputs)
            mse = torch.sum((inputs.actions - x_reconstructed)**2, dim=-1)
            log_px_given_z = -0.5 * (torch.sum(mse, dim=-1)  / (sigma**2) + torch.log(torch.tensor(2 * math.pi * sigma**2, device=z_sample.device)))
            log_pz = -0.5 * torch.sum(z_sample**2, dim=-1)
            log_qz_given_x = -0.5 * torch.sum((z_sample - output.p.mu)**2 / output.p.sigma**2 + torch.log(output.p.sigma**2), dim=-1)
            log_weight = log_px_given_z + log_pz - log_qz_given_x
            log_weights.append(log_weight)
        log_weights = torch.stack(log_weights, dim=0)
        log_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(k, device=z_sample.device))
        return log_likelihood.mean().item()
        #self.logger.log_scalar(log_likelihood.mean().item(), "LL",self.global_step, phase='val')
        


def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")


def get_exp_dir():
    return os.environ['EXP_DIR']


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

def add_log_information(output,logger,phase,step):
    output.log_q_zx = output.q.log_prob(output.z)
    z_mean = output.z.mean(dim=0)
    z_var = output.z.var(dim=0, unbiased=False)
    
    # 주변 분포 로그 가능도 log q(z) 계산
    output.log_q_z = -0.5 * ((output.z - z_mean) ** 2 / z_var + torch.log(z_var) + np.log(2 * np.pi)).sum(dim=1)
    output.mi_estimate = (output.log_q_zx - output.log_q_z)
    logger.log_scalar(output.log_q_z.mean(), "log_q_z", step, phase)
    logger.log_scalar(output.log_q_zx.mean(), "log_q_zx", step, phase)
    logger.log_scalar(output.mi_estimate.mean(), "mi_estimate", step, phase)



if __name__ == '__main__':
    ModelTrainer(args=get_args())

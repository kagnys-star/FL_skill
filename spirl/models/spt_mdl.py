from contextlib import contextmanager
from torch import autograd
import copy
import torch
import torch.nn as nn
import numpy as np
from spirl.components.checkpointer import freeze_modules
from spirl.modules.losses import PenaltyLoss
from spirl.modules.subnetworks import BaseProcessingLSTM
from spirl.utils.general_utils import ParamDict, batch_apply ,AttrDict
from spirl.modules.variational_inference import MultivariateGaussian
from collections import OrderedDict

class MMDEncoder():
    """Skill embedding + prior model for SPIRL algorithm."""
    def __init__(self, params, logger=None):
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self.device = self._hp.device
        self.build_network()


    def _default_hparams(self):
        # put new parameters in here:
        default_dict = ParamDict({
            'use_convs': False,
            'device': None,
            'n_rollout_steps': 10,        # number of decoding steps
            'cond_decode': False,         # if True, conditions decoder on prior inputs
            'batch_size': -1,
            'normalization': 'batch',
            'kernel_type' : 'RBF',
            'penalty_weights' : 1,
            'sigma' : 1,
        })

        # Network size
        default_dict.update({
            'state_dim': 1,             # dimensionality of the state space
            'action_dim': 1,            # dimensionality of the action space
            'nz_enc': 32,               # number of dimensions in encoder-latent space
            'nz_vae': 10,               # number of dimensions in vae-latent space
            'nz_mid': 32,               # number of dimensions for internal feature spaces
            'nz_mid_lstm': 128,         # size of middle LSTM layers
            'n_lstm_layers': 1,         # number of LSTM layers
            'n_processing_layers': 3,   # number of layers in MLPs
        })
        return default_dict

    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        input_size = self._hp.action_dim + self.prior_input_size

        self.global_q = torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2))
        self.global_q.to(self.device)
        freeze_modules([self.global_q])

    def calculate(self, inputs, output, losses):
        """Forward pass of the SPIRL model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        inputs.observations = inputs.actions    # for seamless evaluation
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)

        # run inference
        with autograd.no_grad():
            output.global_q = MultivariateGaussian(self.global_q(inf_input)[:, -1])
            output.z_global = output.global_q.sample()

        n = output.z.size(0)
        nf = float(n)
        half_size = (n * n - n) // 2

        norms_pz = torch.sum(output.z ** 2, dim=1, keepdim=True)
        dotprods_pz = torch.matmul(output.z, output.z.t())
        distances_pz = norms_pz + norms_pz.t() - 2.0 * dotprods_pz

        norms_qz = torch.sum(output.z_global ** 2, dim=1, keepdim=True)
        dotprods_qz = torch.matmul(output.z_global, output.z_global.t())
        distances_qz = norms_qz + norms_qz.t() - 2.0 * dotprods_qz

        dotprods = torch.matmul(output.z_global, output.z.t())
        distances = norms_qz + norms_pz.t() - 2.0 * dotprods
        
        # MMD Loss
        if self._hp.kernel_type == "RBF":
            sigma2_k = torch.topk(distances.view(-1), half_size, largest=False).values[-1]
            sigma2_k += torch.topk(distances_qz.view(-1), half_size, largest=False).values[-1]
            res1 = torch.exp(-distances_qz / (2.0 * sigma2_k))
            res1 += torch.exp(-distances_pz / (2.0 * sigma2_k))
            res1 = res1 * (1.0 - torch.eye(n, device=self.device))
            res1 = torch.sum(res1) / (nf * nf - nf)
            res2 = torch.exp(-distances / (2.0 * sigma2_k))
            res2 = torch.sum(res2) * 2.0 / (nf * nf)
            stat = res1 - res2
            '''
            xx, yy, zz = torch.mm(output.z, output.z.t()), torch.mm(output.z_global, output.z_global.t()), torch.mm(output.z, output.z_global.t())
            rx = (xx.diag().unsqueeze(0).expand_as(xx))
            ry = (yy.diag().unsqueeze(0).expand_as(yy))

            dxx = rx.t() + rx - 2. * xx # Used for A in (1)
            dyy = ry.t() + ry - 2. * yy # Used for B in (1)
            dxy = rx.t() + ry - 2. * zz # Used for C in (1)
            a = self._hp.sigma
            x_1 = torch.exp(-0.5*dxx/a)
            y_1 = torch.exp(-0.5*dyy/a)
            xy_1 = torch.exp(-0.5*dxy/a)
            ''' 
        elif self._hp.kernel_type == "IMQ":
            cbase = 2. * self.latent_dim
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                c = cbase * scale
                res1 = c / (c + distances_qz)
                res1 += c / (c + distances_pz)
                res1 * (1.0 - torch.eye(n, device=self.device))
                res1 = torch.sum(res1) / (nf * nf - nf)
                res2 = c / (c + distances)
                res2 = torch.sum(res2) * 2.0 / (nf * nf)
                stat += res1 - res2

        #losses.MMD = PenaltyLoss(self._hp.penalty_weights)(x_1 + y_1 - 2. * xy_1)
        losses.MMD = PenaltyLoss(self._hp.penalty_weights)(stat)
        if hasattr(losses, 'total'):
            del losses.total
        losses.total = self._compute_total_loss(losses)

        return losses

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def load_weights(self, model_type, weights):        
        if model_type == "global":
            self.global_q.load_state_dict(weights)
            freeze_modules([self.global_q])
        elif model_type == "pre":
            self.pre_q.load_state_dict(weights)
            freeze_modules([self.pre_q])
        else:
            raise ValueError("'{}' not supported!".format(model_type))
        
    @property
    def resolution(self):
        return 64       # return dummy resolution, images are not used by this model

    @property
    def latent_dim(self):
        return self._hp.nz_vae

    @property
    def state_dim(self):
        return self._hp.state_dim

    @property
    def prior_input_size(self):
        return self.state_dim

    @property
    def n_rollout_steps(self):
        return self._hp.n_rollout_steps


    @staticmethod
    def _compute_total_loss(losses):
        total_loss = torch.stack([loss[1].value * loss[1].weight for loss in
                                  filter(lambda x: x[1].weight > 0, losses.items())]).sum()
        return AttrDict(value=total_loss)
    

class MoonEncoder(MMDEncoder):


    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        input_size = self._hp.action_dim + self.prior_input_size

        self.global_q = torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2))
        self.global_q.to(self.device)
        freeze_modules([self.global_q])

        self.pre_q = torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2))
        self.pre_q.to(self.device)
        freeze_modules([self.pre_q])

    def calculate(self, inputs, output, losses):
        inputs.observations = inputs.actions    # for seamless evaluation
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)

        # run inference
        with autograd.no_grad():
            output.global_q = MultivariateGaussian(self.global_q(inf_input)[:, -1])
            output.z_global = output.global_q.sample()
            output.pre_q = MultivariateGaussian(self.pre_q(inf_input)[:, -1])
            output.z_pre = output.pre_q.sample()
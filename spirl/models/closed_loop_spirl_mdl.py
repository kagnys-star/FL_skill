import torch
import torch.nn as nn
import math
from spirl.modules.losses import KLDivLoss, NLL, L2Loss , PenaltyLoss
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import batch_apply, ParamDict
from spirl.utils.pytorch_utils import get_constant_parameter, ResizeSpatial, RemoveSpatial
from spirl.models.skill_prior_mdl import SkillPriorMdl, ImageSkillPriorMdl
from spirl.modules.subnetworks import Predictor, BaseProcessingLSTM, Encoder
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.components.checkpointer import load_by_key, freeze_modules


class ClSPiRLMdl(SkillPriorMdl):
    """SPiRL model with closed-loop low-level skill decoder."""
    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self.log_sigma = get_constant_parameter(0., learnable=False)


    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.prior_input_size
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1])

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.q])
        else:
            super().load_weights_and_freeze()

    @property
    def enc_size(self):
        return self._hp.state_dim


class ImageClSPiRLMdl(ClSPiRLMdl, ImageSkillPriorMdl):
    """SPiRL model with closed-loop decoder that operates on image observations."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'prior_input_res': 32,      # input resolution of prior images
            'encoder_ngf': 8,           # number of feature maps in shallowest level of encoder
            'n_input_frames': 1,        # number of prior input frames
        })
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _build_prior_net(self):
        return ImageSkillPriorMdl._build_prior_net(self)

    def _build_inference_net(self):
        self.img_encoder = nn.Sequential(ResizeSpatial(self._hp.prior_input_res),  # encodes image inputs
                                         Encoder(self._updated_encoder_params()),
                                         RemoveSpatial(),)
        return ClSPiRLMdl._build_inference_net(self)

    def _get_seq_enc(self, inputs):
        # stack input image sequence
        stacked_imgs = torch.cat([inputs.images[:, t:t+inputs.actions.shape[1]]
                                  for t in range(self._hp.n_input_frames)], dim=2)
        # encode stacked seq
        return batch_apply(stacked_imgs, self.img_encoder)

    def _learned_prior_input(self, inputs):
        return ImageSkillPriorMdl._learned_prior_input(self, inputs)

    def _regression_targets(self, inputs):
        return ImageSkillPriorMdl._regression_targets(self, inputs)

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return self.img_encoder(obs)

    @property
    def enc_size(self):
        return self._hp.nz_enc

    @property
    def prior_input_size(self):
        return self.enc_size


class MuloptMdl(ClSPiRLMdl):


    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)


        # learned skill prior net loss
        

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        return losses


    def loss_q(self, model_output):
        losses = AttrDict()
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        losses.total = self._compute_total_loss(losses)
        return losses
    

class DualSPiRLMdl(ClSPiRLMdl):
    def __init__(self, params, logger=None):
        super().__init__(params, logger)

    def build_network(self):
        super().build_network()
        self.global_encoder = self._build_inference_net()
        self.discriminator = self._build_discriminator_net()


    def _build_discriminator_net(self):
        # condition inference on states since decoder is conditioned on states too
        return torch.nn.Sequential(
            nn.Linear(self._hp.nz_vae, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, use_learned_prior=False):
        output = super().forward(inputs,use_learned_prior)
        output.fixed_z = output.q.sample()
        output.d_noise = self.discriminator(output.fixed_z)
        output.d_data = self.discriminator(output.z)
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)

        data_loss = torch.log(model_output.d_data + 1e-8).mean()
        noise_loss = torch.log(1 - model_output.d_noise + 1e-8).mean()
        losses.disc_loss= PenaltyLoss(1.0)(-(data_loss + noise_loss))

        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        # learned skill prior net loss
    
        # Optionally update beta

        return losses

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        mu_1, logsigma_1 = torch.chunk(self.q(inf_input)[:, -1],2,1)
        mu_2, logsigma_2 = torch.chunk(self.global_encoder(inf_input)[:, -1],2,1)
        mu = mu_1 + mu_2
        log_sigma = torch.log(torch.sqrt(logsigma_1.exp()**2 + logsigma_2.exp()**2))
        return MultivariateGaussian(mu, log_sigma)

    def _compute_learned_prior_loss(self, model_output):
        target = model_output.q.detach()
        det_sigma_q = torch.prod(target.sigma, dim=-1,) # B
        det_sigma_p = torch.prod(model_output.q_hat.sigma, dim=-1)  # B

# Compute the Mahalanobis distance term
        delta_mu = target.mu.detach() - model_output.q_hat.mu  # B x D
        mahalanobis_term = torch.sum((delta_mu**2) / model_output.q_hat.sigma, dim=-1)  # B

# Compute the log ratio of determinants
        log_det_ratio = torch.log(det_sigma_p / det_sigma_q + 1e-8)  # B

# Softened Reverse KL formula
        softened_kl = 2 * torch.log(
            1 + torch.exp(0.5 * (log_det_ratio - mahalanobis_term))
        )
        loss = PenaltyLoss(breakdown=0)(softened_kl)
        loss.breakdown = torch.stack([chunk.mean() for chunk in torch.chunk(loss.breakdown, self._hp.n_prior_nets)])
        loss.weight =  self._hp.q_hat_weight
        return loss


class WAESPiRLMdl(ClSPiRLMdl):
    def loss(self, model_output, inputs):
        kernel_type = "IMQ"
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))

        n = model_output.z.size(0)
        nf = float(n)
        half_size = (n * n - n) // 2

        norms_pz = torch.sum(model_output.z ** 2, dim=1, keepdim=True)
        dotprods_pz = torch.matmul(model_output.z, model_output.z.t())
        distances_pz = norms_pz + norms_pz.t() - 2.0 * dotprods_pz

        norms_qz = torch.sum(model_output.z_q ** 2, dim=1, keepdim=True)
        dotprods_qz = torch.matmul(model_output.z_q, model_output.z_q.t())
        distances_qz = norms_qz + norms_qz.t() - 2.0 * dotprods_qz

        dotprods = torch.matmul(model_output.z_q, model_output.z.t())
        distances = norms_qz + norms_pz.t() - 2.0 * dotprods
        
        # MMD Loss
        if kernel_type == "RBF":
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
        elif kernel_type == "IMQ":
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
        # KL loss
        losses.kl_loss = PenaltyLoss(self.beta)(stat)

        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        # learned skill prior net loss
    
        # Optionally update beta

        return losses


class SP_WAE_iRLMdl(ClSPiRLMdl):


    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)

        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        # learned skill prior net loss
    
        # Optionally update beta

        return losses
    
    def _compute_learned_prior_loss(self, model_output):
        if self._hp.nll_prior_train:
            loss = NLL(breakdown=0)(model_output.q_hat, model_output.z_q.detach())
        else:
            
            target = model_output.q.detach()
            det_sigma_q = torch.prod(target.sigma, dim=-1,) # B
            det_sigma_p = torch.prod(model_output.q_hat.sigma, dim=-1)  # B

    # Compute the Mahalanobis distance term
            delta_mu = target.mu.detach() - model_output.q_hat.mu  # B x D
            mahalanobis_term = torch.sum((delta_mu**2) / model_output.q_hat.sigma, dim=-1)  # B

    # Compute the log ratio of determinants
            log_det_ratio = torch.log(det_sigma_p / det_sigma_q + 1e-8)  # B

    # Softened Reverse KL formula
            softened_kl = 2 * torch.log(
                1 + torch.exp(0.5 * (log_det_ratio - mahalanobis_term))
            )
            loss = PenaltyLoss(breakdown=0)(softened_kl)
        loss.breakdown = torch.stack([chunk.mean() for chunk in torch.chunk(loss.breakdown, self._hp.n_prior_nets)])
        loss.weight =  self._hp.q_hat_weight
        return loss
    


class TCSPiRLMdl(ClSPiRLMdl):
    def __init__(self, params, logger=None):
        super().__init__(params, logger)
        self.dataset_size = 1

    def loss(self, model_output, inputs):

        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))
        
        losses.kl = KLDivLoss(self.beta)(model_output.q, model_output.p)

        ####



        losses.tc = self._compute_TC_loss(model_output)

        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        # learned skill prior net loss
    
        # Optionally update beta

        return losses
    

    def _compute_TC_loss(self, model_output):
        batch_size, latent_dim = model_output.z.size()
        
        _logqz = model_output.q.keep_log_prob(model_output.z)
        
        logqz_prodmarginals = 0
        for i in range(latent_dim):
            log_q_z_i_given_x = _logqz[:, i]  # Extract individual z_i
            log_q_z_i = torch.logsumexp(log_q_z_i_given_x, dim=0) - math.log(batch_size * self.dataset_size)
            logqz_prodmarginals += log_q_z_i
        logqz = torch.logsumexp(_logqz.sum(-1), dim=0)- math.log(batch_size * self.dataset_size)
        
        '''
        val = model_output.z.unsqueeze(1)
        mu =  model_output.q.mu.unsqueeze(0)
        sigma = model_output.q.sigma.unsqueeze(0)
        log_sigma = model_output.q.log_sigma.unsqueeze(0)
        _logqz = -1 * ((val - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi))
        logqz_prodmarginals = (self.logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * self.dataset_size)).sum(1)
        logqz = (self.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * self.dataset_size))
        '''
        return PenaltyLoss(1.0)(logqz - logqz_prodmarginals)


    def logsumexp( self ,value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                        dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + math.log(sum_exp)


class NCESPiRLMdl(ClSPiRLMdl):
    #help global encoder encorage to know which model
    def __init__(self, params, logger=None):
        super().__init__(params, logger)

    def build_network(self):
        super().build_network()
        self.global_encoder = self._build_inference_net()
        self.discriminator = self._build_discriminator_net()


    def _build_discriminator_net(self):
        # condition inference on states since decoder is conditioned on states too
        return torch.nn.Sequential(
            nn.Linear(self._hp.nz_vae, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, inputs, use_learned_prior=False):
        output = super().forward(inputs,use_learned_prior)
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        z_noise = MultivariateGaussian(self.global_encoder(inf_input)[:, -1])
        output.d_noise = self.discriminator(output.z)
        output.d_data = self.discriminator(z_noise.sample().detach())
        return output



    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)
        # learned skill prior net loss
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        data_loss = torch.log(model_output.d_data + 1e-8).mean()
        noise_loss = torch.log(1 - model_output.d_noise + 1e-8).mean()
        losses.disc_loss= PenaltyLoss(1.0)(-(data_loss + noise_loss))

        losses.total = self._compute_total_loss(losses)
        return losses



class DENCESPiRLMdl(ClSPiRLMdl):
    #help global encoder encorage to know which model
    def __init__(self, params, logger=None):
        super().__init__(params, logger)

    def build_network(self):
        super().build_network()
        self.discriminator = self._build_discriminator_net()


    def _build_discriminator_net(self):
        # condition inference on states since decoder is conditioned on states too
        return torch.nn.Sequential(
            nn.Linear(self._hp.nz_vae, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, inputs, use_learned_prior=False):
        output = super().forward(inputs,use_learned_prior)
        output.d_noise = self.discriminator(output.z_q)
        output.d_data = self.discriminator(output.z)
        return output



    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)

        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)
        data_loss = torch.log(model_output.d_data + 1e-8).mean()
        noise_loss = torch.log(1 - model_output.d_noise + 1e-8).mean()
        losses.disc_loss= PenaltyLoss(1.0)(-(data_loss + noise_loss))

        losses.total = self._compute_total_loss(losses)
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        # learned skill prior net loss
    
        # Optionally update beta

        return losses
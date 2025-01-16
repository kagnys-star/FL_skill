import torch
import torch.nn as nn
import copy 
from spirl.modules.losses import KLDivLoss, NLL ,CEELoss ,L2Loss , PenaltyLoss ,BCELoss
from torch.autograd import Variable
from spirl.utils.general_utils import batch_apply, ParamDict, AttrDict
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.modules.recurrent_modules import BaseProcessingLSTM , DiscProcessingLSTM
from spirl.modules.variational_inference import get_fixed_prior

class dvaeMdl(ClSPiRLMdl):
    """SPiRL model with closed-loop low-level skill decoder."""
    def build_network(self):
        super().build_network()
        self.discriminator = self._build_discriminator_net()

    def _build_discriminator_net(self):
        # condition inference on states since decoder is conditioned on states too
        return torch.nn.Sequential(
            DiscProcessingLSTM(self._hp, in_dim=self._hp.action_dim , out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, 1)
        )
    

    def forward(self, inputs, use_learned_prior=False , mode="vae"):
        """Forward pass of the SPIRL model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()
        inputs.observations = inputs.actions    # for seamless evaluation
        inputs.ones_label=Variable(torch.ones(inputs.actions.shape[0],1)).to(self.device)
        inputs.zeros_label=Variable(torch.zeros(inputs.actions.shape[0],1)).to(self.device)
        inputs.zeros_label1=Variable(torch.zeros(inputs.actions.shape[0],1)).to(self.device)

        # run inference
        output.q = self._run_inference(inputs)

        # compute (fixed) prior
        output.p = get_fixed_prior(output.q)


        # infer learned skill prior
        output.q_hat = self.compute_learned_prior(self._learned_prior_input(inputs))
        if use_learned_prior:
            output.p = output.q_hat     # use output of learned skill prior for sampling

        # sample latent variable
        output.z = output.p.sample() if self._sample_prior else output.q.sample()
        output.z_q = output.z.clone() if not self._sample_prior else output.q.sample()   # for loss computation
        output.z_p = output.q_hat.sample()
        # decode
        assert self._regression_targets(inputs).shape[1] == self._hp.n_rollout_steps
        output.reconstruction = self.decode(output.z,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps,
                                            inputs=inputs)
        output.s_reconstruction = self.decode(output.z_q,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps,
                                            inputs=inputs)
        self._run_discriminator( output, inputs)
        return output
    

    def loss(self, model_output, inputs , mode="vae" ):
        """Loss computation of the SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """
        losses = AttrDict()
        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = L2Loss(self._hp.reconstruction_mse_weight) \
            (model_output.reconstruction,
             self._regression_targets(inputs))
        # KL loss
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)
        # learned skill prior net loss
        #losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        
        losses.gan_loss = self._compute_discriminator_loss(model_output, inputs)
        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)
        losses.total = self._compute_total_loss(losses)
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        super()._log_outputs(model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs)


    def _run_discriminator(self, output, inputs):
        # run inference with state sequence conditioning
        m = nn.Sigmoid()
        output.disc_r =m(self.discriminator(inputs.actions).squeeze(-1))
        output.disc_f = m(self.discriminator(output.reconstruction).squeeze(-1))
        output.disc_s = m(self.discriminator(output.s_reconstruction).squeeze(-1))

    def _compute_discriminator_loss(self, output, inputs):
        losses = AttrDict()
        losses.disc_r_loss = BCELoss(1.0)(output.disc_r, inputs.ones_label)
        losses.disc_f_loss = BCELoss(1.0)(output.disc_f, inputs.zeros_label)
        losses.disc_s_loss = BCELoss(1.0)(output.disc_s, inputs.zeros_label1)
        total_loss = torch.stack([loss[1].value * loss[1].weight for loss in
                                  filter(lambda x: x[1].weight > 0, losses.items())]).sum()
        return PenaltyLoss(1.0)(total_loss)

    def loss_q(self, model_output):
        losses = AttrDict()
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)
        losses.total = self._compute_total_loss(losses)
        return losses
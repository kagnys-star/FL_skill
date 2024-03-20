import torch
import torch.nn as nn
from spirl.modules.losses import KLDivLoss, NLL ,CEELoss ,L2Loss
from spirl.utils.general_utils import batch_apply, ParamDict, AttrDict
from spirl.utils.pytorch_utils import get_constant_parameter, ResizeSpatial, RemoveSpatial
from spirl.models.skill_prior_mdl import SkillPriorMdl, ImageSkillPriorMdl
from spirl.modules.subnetworks import Predictor, BaseProcessingLSTM, Encoder
from spirl.modules.variational_inference import MultivariateGaussian,get_fixed_prior,Gaussian
from spirl.components.checkpointer import load_by_key, freeze_modules
from spirl.utils.eval_utils import mse

class FlSPiRLMdl(SkillPriorMdl):
    """SPiRL model with closed-loop low-level skill decoder."""
    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.classifier = self._build_class_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self.log_sigma = get_constant_parameter(0., learnable=False)

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.prior_input_size
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )
    def forward(self, inputs, use_learned_prior=False):
        output = AttrDict()
        inputs.observations = inputs.actions    # for seamless evaluation

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
        output.q_reconstruction = self.decode(output.z_p,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps,
                                            inputs=inputs)
        output.skill = self.classifier(output.z)

        return output
    

    def loss(self, model_output, inputs):
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
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.skill_loss = CEELoss(0.5)(model_output.skill, inputs.tasks)
        losses.total = self._compute_total_loss(losses)

        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        """Optionally visualizes outputs of SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg losses: output of SPIRL model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        self._logger.log_scalar(self.beta, "beta", step, phase)
        self._logger.log_scalar(model_output.q.mu.mean(), "q_mu", step, phase)
        self._logger.log_scalar(model_output.q_hat.mu.mean(), "q_hat_mu", step, phase)
        self._logger.log_scalar(model_output.q.sigma.mean(), "q_sigma", step, phase)
        self._logger.log_scalar(model_output.q_hat.sigma.mean(), "q_hat_sigma", step, phase)
        self._logger.log_scalar(mse(model_output.reconstruction,(model_output.q_reconstruction )), "prior_mse", step, phase)

        # log videos/gifs in tensorboard
        if log_images:
            print('{} {}: logging videos'.format(phase, step))
            self._logger.visualize(model_output, inputs, losses, step, phase, logger, **logging_kwargs)




    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

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

    def _build_class_net(self):
        input_size = self._hp.nz_vae
        return torch.nn.Sequential(
            torch.nn.Linear(input_size,self._hp.nz_enc),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self._hp.nz_enc, 7),
        )



    @property
    def enc_size(self):
        return self._hp.state_dim

import torch
import torch.nn as nn
from spirl.modules.losses import KLDivLoss, NLL ,CEELoss ,L2Loss , PenaltyLoss
from spirl.utils.general_utils import batch_apply, ParamDict, AttrDict
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.modules.subnetworks import LinearClassifier

class INFOSPiRLMdl(ClSPiRLMdl):
    """SPiRL model with closed-loop low-level skill decoder."""
    def build_network(self):
        super().build_network()
        if self._hp.linear_condition == True:
            self.c = LinearClassifier(self.enc_size + self._hp.nz_vae, self._hp.tasks)
        else: 
            self.c = LinearClassifier(self._hp.nz_vae , self._hp.tasks)
    
    def _default_hparams(self):
        default_dict = ParamDict({
            'label_weights': 1.0,
            "tasks" : 4,
            "linear_condition" : True,
        })
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def forward(self, inputs, use_learned_prior=False):
        output = super().forward(inputs, use_learned_prior)
        if self._hp.linear_condition == True:
            seq_enc = self._get_seq_enc(inputs)
            condition_inputs = torch.cat((seq_enc[:,0], output.z), dim=-1)
            output.labels = self.c(condition_inputs)
        else:
            output.labels = self.c(output.z)
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
        losses.lablel_loss = CEELoss(self._hp.label_weights)(model_output.labels, inputs.labels)
        losses.unit_norm = PenaltyLoss(1.0)(torch.sum((torch.norm(self.c.linear.weight  , p=2, dim=1) - 1) ** 2))
        losses.total = self._compute_total_loss(losses)
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        super()._log_outputs(model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs)
        predictions = torch.argmax(model_output.labels, dim=1)
        target = torch.argmax(inputs.labels, dim=1)
        self._logger.log_scalar((predictions == target).float().mean().item(), "accuracy", step, phase)
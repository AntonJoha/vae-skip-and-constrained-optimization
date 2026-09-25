import logging

import torch

from .kl_reg import Model as RegModel

log = logging.getLogger(__name__)


class Model(RegModel):
    def __init__(self, config):
        super().__init__(config)
        self.beta = float(self.config.beta)
        total_epochs = self.config.tuning_epochs if self.config.tune else self.config.epochs
        warmup_fraction = min(
            1.0, max(0.0, float(getattr(self.config, "vae_kl_warmup_fraction", 0.3)))
        )
        if warmup_fraction == 0.0:
            self.kl_warmup_epochs = 0
        else:
            self.kl_warmup_epochs = max(1, int(total_epochs * warmup_fraction))
        self.kl_weight = 0.0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.kl_warmup_epochs == 0:
            self.kl_weight = self.beta
        else:
            warmup_denom = max(1, self.kl_warmup_epochs - 1)
            warmup_progress = min(1.0, float(epoch) / float(warmup_denom))
            self.kl_weight = self.beta * warmup_progress
        log.info(
            "Epoch %d: KL warmup weight set to %.4f",
            epoch,
            self.kl_weight,
        )

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False
        )

        rec, kl_loss = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=prior_list,
            combined_posterior_list=combined_posterior_list,
        )
        layered_kl = self._layered_kl(prior_list, combined_posterior_list)
        loss = rec + self.kl_weight * kl_loss

        loss.backward()
        optimizer.step()

        self.last_train_metrics = {
            "posterior_recon": float(rec.detach()),
            "posterior_kl": float(layered_kl.detach().sum()),
            "layered_kl": layered_kl.detach(),
        }

        return float(loss.detach())

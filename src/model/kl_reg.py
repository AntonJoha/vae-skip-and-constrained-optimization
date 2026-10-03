import logging

from experiments.util import SeriesConfig

from .kl_base import KLBaseModel

log = logging.getLogger(__name__)

TDLGMConfig = SeriesConfig


class Model(KLBaseModel):
    state_dropout = 0.0
    layer_dropout = 0.0
    logvar_clamp = None
    combine_gaussian = False
    posterior_reduce = "last"
    reverse_posterior_list = False

    def __init__(self, config):
        super().__init__(config)
        self.lambda_ = 1.0
        self.lambda_lr = 1e-3
        self.kl_target = float(self.config.beta)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = 1 / (self.epoch**self.config.beta) if self.epoch > 0 else 1.0
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def train_step(
        self, x, y, optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False
        )

        rec, _ = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=prior_list,
            combined_posterior_list=combined_posterior_list,
        )
        layered_kl = self._layered_kl(prior_list, combined_posterior_list)
        loss = rec + (self.kl_target - layered_kl).pow(2).mean()

        loss.backward()
        optimizer.step()
        return float(loss.detach())

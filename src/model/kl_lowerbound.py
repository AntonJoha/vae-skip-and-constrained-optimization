import logging

import torch

from experiments.util import SeriesConfig

from .kl_base import KLBaseModel

log = logging.getLogger(__name__)

TDLGMConfig = SeriesConfig


class Model(KLBaseModel):
    state_dropout = 0.0
    layer_dropout = 0.0
    logvar_clamp = None
    combine_gaussian = False
    posterior_reduce = "mean"
    reverse_posterior_list = False

    def __init__(self, config):
        super().__init__(config)
        self.lambda_ = 0.0
        self.lambda_lr = 1e-4
        self.kl_penalty = 1.0
        self.kl_target = 0.5
        self.rho = self.config.rho
        self.rho_scaler = float(self.config.rho_scaler)
        self.reduction_threshold = self.config.reduction_threshold
        self.old_violation = torch.inf

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = float(self.config.beta)
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def _layer_target(self, layered_kl: torch.Tensor) -> torch.Tensor:
        return layered_kl.new_full(
            layered_kl.shape,
            float(self.kl_target),
        )

    def _update_dual_variables(self, residual: torch.Tensor) -> None:
        constrain_violation = max(0.0, abs(residual.item()))
        if constrain_violation > self.reduction_threshold * self.old_violation:
            self.rho *= self.rho_scaler
        self.old_violation = constrain_violation
        self.lambda_ = min(max(0.0, self.lambda_ + self.rho * residual.item()), 50)

    def outer_train_step(self, dataloader):
        expected_kl = None
        expected_wasserstein = None
        latent_mean_diff = None
        latent_logvar_diff = None

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            kl = self.get_layered_kl(x, y)
            wasserstein = self.get_layered_wasserstein(x, y)
            mean_diff, logvar_diff = self.get_mean_logvar_diff(x, y)
            if expected_kl is None:
                expected_kl = kl
            else:
                expected_kl += kl
            if expected_wasserstein is None:
                expected_wasserstein = wasserstein
            else:
                expected_wasserstein += wasserstein
            if latent_mean_diff is None:
                latent_mean_diff = mean_diff
            else:
                latent_mean_diff += mean_diff
            if latent_logvar_diff is None:
                latent_logvar_diff = logvar_diff
            else:
                latent_logvar_diff += logvar_diff

        expected_kl /= len(dataloader)
        expected_wasserstein /= len(dataloader)
        latent_mean_diff /= len(dataloader)
        latent_logvar_diff /= len(dataloader)

        expected_kl = expected_kl.sum()
        residual = self.kl_target - expected_kl
        log.info(
            "Outer step: expected KL=%.4f, residual=%.4f, lambda=%.4f, rho=%.4f, Expected Wasserstein=%.4f, latent mean diff=%.4f, latent logvar diff=%.4f",
            expected_kl.item(),
            residual.mean().item(),
            self.lambda_,
            self.rho,
            expected_wasserstein.mean().item(),
            latent_mean_diff.mean().item(),
            latent_logvar_diff.mean().item(),
        )

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False
        )
        _, prior_kl = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=prior_list,
            combined_posterior_list=[t.detach() for t in combined_posterior_list],
        )
        prior_kl.backward()
        optimizer.step()

        optimizer.zero_grad()
        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False
        )
        rec, kl = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=[t.detach() for t in prior_list],
            combined_posterior_list=combined_posterior_list,
        )

        residual = self.kl_target - kl
        shifted = self.lambda_ + self.rho * residual
        al_penalty = (
            torch.clamp(shifted, min=0.0) ** 2 - self.lambda_**2
        ) / (2.0 * self.rho)
        loss = rec + al_penalty
        loss.backward()
        optimizer.step()
        self._update_dual_variables(residual.detach())

        return float(loss.detach())

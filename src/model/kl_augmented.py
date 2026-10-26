import logging

import torch

from experiments.util import SeriesConfig

from .kl_base import KLBaseModel

log = logging.getLogger(__name__)

TDLGMConfig = SeriesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(KLBaseModel):
    state_dropout = 0.2
    layer_dropout = 0.2
    logvar_clamp = (-10.0, 2.0)
    combine_gaussian = False
    posterior_reduce = "mean"
    reverse_posterior_list = True

    def __init__(self, config):
        super().__init__(config)
        self.kl_target = float(self.config.beta)
        self.lambda_lr = float(getattr(config, "lr_lambda", self.lambda_lr))
        self.rho_lr = float(getattr(config, "rho_lr", self.rho_lr))
        self.lambda_ = torch.zeros(config.layers).to(device)
        self.old_violation = torch.full((config.layers,), torch.inf)

        self.lambda_min = torch.zeros(config.layers) - 50
        self.lambda_max = torch.zeros(config.layers) + 50
        self.lambda_min = self.lambda_min.to(device)
        self.lambda_max = self.lambda_max.to(device)

        if config.set_lambda is not None:
            self.lambda_ += config.set_lambda
            for i in range(config.layers):
                self.lambda_[i] = self.lambda_[i]*config.lambda_scaler**(i)

    def _layer_target(self, layered_kl: torch.Tensor) -> torch.Tensor:
        return self._layerwise_budget_target(layered_kl)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def outer_train_step(self, dataloader):
        # lambda and rho are updated every batch in train_step
        return None

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        _inner_optimizer: torch.optim.Optimizer | None = None,
        inner_optimizer=None,
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
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
        residual = layered_kl - self._layer_target(layered_kl)

        kl_constraint = (
            self.lambda_.to(residual.device) * residual
            + 0.5 * self.rho * residual.pow(2)
        ).sum()
        loss = rec + kl_constraint
        loss.backward()
        optimizer.step()

        dual_lr = optimizer.param_groups[0]["lr"] / 100
        self.lambda_lr = dual_lr
        self.rho_lr = dual_lr
        detached_residual = residual.detach()
        self.rho = max(
            1e-8, self.rho + self.rho_lr * float(detached_residual.abs().mean().item())
        )
        if self.config.set_lambda is None:
            self.lambda_ = torch.clamp(
                self.lambda_.to(detached_residual.device)
                + self.lambda_lr * detached_residual,
                self.lambda_min,
                self.lambda_max,
            )

        return float(loss.detach())

    def print_gradients(self, x: torch.Tensor, y: torch.Tensor):

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
        residual = layered_kl - self._layer_target(layered_kl)

        kl_constraint = (
            self.lambda_.to(residual.device) * residual
            + 0.5 * self.rho * residual.pow(2)
        ).sum()

        grad_info = self.gradient_diagnostics(
            {
                "reconstruction": rec,
                "kl_constraint": kl_constraint,
                "full_loss": rec + kl_constraint,
            }
        )

        log.info("Scaled gradient norms:")
        for loss_name, groups in grad_info.items():
            log.info("  %s:", loss_name)
            for group_name, norm in groups.items():
                log.info("    %-20s %.6e", group_name, norm)
        log.info(
            "  reconstruction=%.6f kl_constraint=%.6f",
            rec.item(),
            kl_constraint.item(),
        )



        cosine_similarity = self.gradient_cosine_similarity(rec, kl_constraint)
        log.info("Cosine reconstruction, kl_constraint:")
        for group_name, similarity in cosine_similarity.items():
            log.info("    %-20s %.6f", group_name, similarity)

        prior_mean, prior_logvar, *_ = self._latent_pass(x, y=None, prior=True)
        """
        log.info(
            "  posterior mean=%.6f logvar=%.6f prior mean=%.6f prior logvar=%.6f",
            mean[0][0][0].item(),
            logvar[0][0][0].item(),
            prior_mean[0][0][0].item(),
            prior_logvar[0][0][0].item(),
        )
        """

    def inner_parameters(self):
        return None

    def middle_parameters(self):
        return (
            list(self.prior_state.parameters())
            + list(self.prior_layers.parameters())
            + list(self.to_output.parameters())
            + list(self.posterior_state.parameters())
            + list(self.posterior_layers.parameters())
        )

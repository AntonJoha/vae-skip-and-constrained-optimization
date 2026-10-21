import logging

import torch

from experiments.util import SeriesConfig

from .kl_base import KLBaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)

TDLGMConfig = SeriesConfig


class Model(KLBaseModel):
    state_dropout = 0.2
    layer_dropout = 0.2
    logvar_clamp = (-10.0, 2.0)
    combine_gaussian = False
    posterior_reduce = "mean"
    reverse_posterior_list = False

    def __init__(self, config):
        super().__init__(config)
        self.lambda_ = 0.0
        self.lambda_lr = float(getattr(config, "lr_lambda", self.lambda_lr))
        self.rho_lr = float(getattr(config, "rho_lr", self.rho_lr))
        self.kl_penalty = 1.0
        self.kl_target = 0.5
        self.rho = self.config.rho
        self.rho_scaler = float(self.config.rho_scaler)
        self.reduction_threshold = self.config.reduction_threshold
        self.lambda_lower = torch.zeros(config.layers).to(device)
        self.lambda_upper = torch.zeros(config.layers).to(device)

        self.lambda_min = torch.zeros(config.layers)
        if config.lambda_min > 0:
            self.lambda_min += config.lambda_min

        self.lambda_max = torch.zeros(config.layers) + config.lambda_max
        self.lambda_min = self.lambda_min.to(device)
        self.lambda_max = self.lambda_max.to(device)
        self.old_violation = torch.full((config.layers,), torch.inf)

        self.spread = 0.2
        log.info("RHO: %f, scaler: %f", self.rho, self.rho_scaler)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = float(self.config.beta)
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def _layer_target(self, layered_kl: torch.Tensor) -> torch.Tensor:
        return self._layerwise_budget_target(layered_kl)

    def outer_train_step(self, dataloader):
        # lambda and rho are updated every batch in train_step
        return None

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer | None,
        _inner_optimizer: torch.optim.Optimizer | None = None,
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
        residual_lower = self._lower_layer_target(layered_kl) - layered_kl

        shifted = self.lambda_lower + self.rho * residual_lower
        al_penalty_lower = (
            torch.clamp(shifted, min=0.0) ** 2 - self.lambda_lower**2
        ) / (2.0 * self.rho)

        residual_upper = layered_kl - self._upper_layer_target(layered_kl)
        lambda_upper = self.lambda_upper.to(residual_upper.device)
        shifted_upper = lambda_upper + self.rho * residual_upper
        al_penalty_upper = (
            torch.clamp(shifted_upper, min=0.0) ** 2 - lambda_upper**2
        ) / (2.0 * self.rho)

        al_penalty = al_penalty_lower.sum() + al_penalty_upper.sum()

        loss = rec + al_penalty.sum()

        loss.backward()
        optimizer.step()

        dual_lr = optimizer.param_groups[0]["lr"] / 100
        self.lambda_lr = dual_lr
        self.rho_lr = dual_lr
        res_lower = residual_lower.detach()
        res_upper = residual_upper.detach()
        self.lambda_lower = torch.clamp(
            self.lambda_lower.to(res_lower.device) + self.lambda_lr * res_lower,
            min=0,
            max=50,
        )
        self.lambda_upper = torch.clamp(
            self.lambda_upper.to(res_upper.device) + self.lambda_lr * res_upper,
            min=0,
            max=50,
        )
        violation = torch.max(
            torch.clamp(res_lower, min=0.0), torch.clamp(res_upper, min=0.0)
        )
        self.rho = max(1e-8, self.rho + self.rho_lr * float(violation.mean().item()))

        return float(loss.detach())

    def inner_parameters(self):
        return None

    def middle_parameters(self):
        return list(self.parameters())

    def print_gradients(self, x: torch.Tensor, y: torch.Tensor):

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

        grad_info = self.gradient_diagnostics(
            {
                "prior_kl": prior_kl,
            }
        )

        log.info("Inner step scaled gradient norms:")
        for loss_name, groups in grad_info.items():
            log.info("  %s:", loss_name)
            for group_name, norm in groups.items():
                log.info("    %-20s %.6e", group_name, norm)

        grad_info = self.gradient_diagnostics(
            {
                "prior_kl": prior_kl,
            }
        )

        log.info("Inner step scaled gradient norms:")
        for loss_name, groups in grad_info.items():
            log.info("  %s:", loss_name)
            for group_name, norm in groups.items():
                log.info("    %-20s %.6e", group_name, norm)

        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False
        )
        rec, _ = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=[t.detach() for t in prior_list],
            combined_posterior_list=combined_posterior_list,
        )

        layered_kl = self._layered_kl(prior_list, combined_posterior_list)
        residual_lower = self._lower_layer_target(layered_kl) - layered_kl

        lambda_lower = self.lambda_lower.to(residual_lower.device)
        shifted = lambda_lower + self.rho * residual_lower
        al_penalty_lower = (torch.clamp(shifted, min=0.0) ** 2 - lambda_lower**2) / (
            2.0 * self.rho
        )

        residual_upper = layered_kl - self._upper_layer_target(layered_kl)
        lambda_upper = self.lambda_upper.to(residual_upper.device)
        shifted_upper = lambda_upper + self.rho * residual_upper
        al_penalty_upper = (
            torch.clamp(shifted_upper, min=0.0) ** 2 - lambda_upper**2
        ) / (2.0 * self.rho)

        al_penalty = al_penalty_lower.sum() + al_penalty_upper.sum()

        grad_info = self.gradient_diagnostics(
            {
                "reconstruction": rec,
                "al_penalty": al_penalty,
                "full_loss": rec + al_penalty,
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
            al_penalty.item(),
        )

        cosine_similarity = self.gradient_cosine_similarity(rec, al_penalty)

        log.info(
            "  Cosine reconstruction, al_penalty:",
        )
        for group_name, similarity in cosine_similarity.items():
            log.info("    %-20s %.6f", group_name, similarity)

        cosine_similarity = self.gradient_cosine_similarity(prior_kl, al_penalty)
        log.info(
            "  Cosine prior_kl, al_penalty:",
        )
        for group_name, similarity in cosine_similarity.items():
            log.info("    %-20s %.6f", group_name, similarity)

        cosine_similarity = self.gradient_cosine_similarity(prior_kl, rec)
        log.info(
            "  Cosine prior_kl, reconstruction:",
        )
        for group_name, similarity in cosine_similarity.items():
            log.info("    %-20s %.6f", group_name, similarity)

        prior_mean, prior_logvar, *_ = self._latent_pass(x, y=None, prior=True)
        log.info(
            "  posterior mean=%.6f logvar=%.6f prior mean=%.6f prior logvar=%.6f",
            mean[0][0][0].item(),
            logvar[0][0][0].item(),
            prior_mean[0][0][0].item(),
            prior_logvar[0][0][0].item(),
        )

    def _upper_layer_target(self, layered_kl):
        layered_target = self._layer_target(layered_kl)
        layered_target += layered_target * self.spread
        return layered_target

    def _lower_layer_target(self, layered_kl):
        layered_target = self._layer_target(layered_kl)
        layered_target -= layered_target * self.spread
        return layered_target

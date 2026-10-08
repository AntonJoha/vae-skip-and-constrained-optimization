import logging

import torch

from experiments.util import SeriesConfig

from .kl_base import KLBaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.lambda_lower = torch.zeros(config.layers)
        self.lambda_upper = torch.zeros(config.layers)

        self.lambda_min = torch.zeros(config.layers)
        if config.lambda_min > 0:
            self.lambda_min += config.lambda_min

        self.lambda_max = torch.zeros(config.layers) + config.lambda_max
        self.lambda_min = self.lambda_min.to(device)
        self.lambda_max = self.lambda_max.to(device)
        self.old_violation = torch.full((config.layers,), torch.inf)

        self.spread = 0.2

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = float(self.config.beta)
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def _layer_target(self, layered_kl: torch.Tensor) -> torch.Tensor:
        return layered_kl.new_full(
            layered_kl.shape,
            float(self.kl_target),
        )

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

        residual_lower = self._lower_layer_target(expected_kl) - expected_kl
        constrain_violation_lower = torch.clamp(residual_lower, min=0.0)
        

        self.lambda_lower = torch.clamp(
                self.lambda_lower.to(residual_lower.device) + self.rho * residual_lower, min=0, max=50
        )

        residual_upper = expected_kl - self._upper_layer_target(expected_kl)
        constrain_violation_upper = torch.clamp(residual_upper, min=0.0)
        self.lambda_upper = torch.clamp(
                self.lambda_upper.to(residual_upper.device) + self.rho * residual_upper, min=0, max=50
        )
        
        constrain_violation = torch.max(constrain_violation_lower, constrain_violation_upper) # max violation??? Or should we sum them?

        if torch.any(
            constrain_violation
            > self.reduction_threshold
            * self.old_violation.to(constrain_violation.device)
        ).item():
            self.rho *= self.rho_scaler
        self.old_violation = constrain_violation
        
        residual = residual_lower + residual_upper # should we sum them? or max?
        log.info(
            "Outer step: expected KL=%.4f, residual=%.4f, lambda lower =%s, lambda upper =%s, rho=%.4f, Expected Wasserstein=%.4f, latent mean diff=%.4f, latent logvar diff=%.4f",
            expected_kl.mean().item(),
            residual.mean().item(),
            self.lambda_lower.cpu().numpy().tolist(),
            self.lambda_upper.cpu().numpy().tolist(),
            self.rho,
            expected_wasserstein.mean().item(),
            latent_mean_diff.mean().item(),
            latent_logvar_diff.mean().item(),
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        middle_optimizer: torch.optim.Optimizer,
        _inner_optimizer = None,
    ) -> float:
        self.train()

        middle_optimizer.zero_grad()
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
        residual_lower= self._lower_layer_target(layered_kl) - layered_kl

        lambda_lower = self.lambda_lower.to(residual_lower.device)
        shifted = lambda_lower + self.rho * residual_lower
        al_penalty_lower = (torch.clamp(shifted, min=0.0) ** 2 - lambda_lower**2) / (
            2.0 * self.rho
        )

        residual_upper= layered_kl - self._upper_layer_target(layered_kl)
        lambda_upper = self.lambda_upper.to(residual_upper.device)
        shifted_upper = lambda_upper + self.rho * residual_upper
        al_penalty_upper = (torch.clamp(shifted_upper, min=0.0) ** 2 - lambda_upper**2) / (
            2.0 * self.rho
        )


        al_penalty = al_penalty_lower.sum() + al_penalty_upper.sum()


        loss = rec + al_penalty.sum()

        loss.backward()
        middle_optimizer.step()

        return float(loss.detach())

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
        residual_lower= self._lower_layer_target(layered_kl) - layered_kl

        lambda_lower = self.lambda_lower.to(residual_lower.device)
        shifted = lambda_lower + self.rho * residual_lower
        al_penalty_lower = (torch.clamp(shifted, min=0.0) ** 2 - lambda_lower**2) / (
            2.0 * self.rho
        )

        residual_upper= layered_kl - self._upper_layer_target(layered_kl)
        lambda_upper = self.lambda_upper.to(residual_upper.device)
        shifted_upper = lambda_upper + self.rho * residual_upper
        al_penalty_upper = (torch.clamp(shifted_upper, min=0.0) ** 2 - lambda_upper**2) / (
            2.0 * self.rho
        )


        al_penalty = al_penalty_lower.sum() + al_penalty_upper.sum()


        grad_info = self.gradient_diagnostics(
            {
                "reconstruction": rec,
                "al_penalty": al_penalty,
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
        layered_target += layered_target*self.spread
        return layered_target

    def _lower_layer_target(self, layered_kl):
        layered_target = self._layer_target(layered_kl) 
        layered_target -= layered_target*self.spread
        return layered_target


















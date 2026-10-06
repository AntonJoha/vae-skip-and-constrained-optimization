import logging

import torch

from experiments.util import SeriesConfig

from .kl_base import KLBaseModel

log = logging.getLogger(__name__)

TDLGMConfig = SeriesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.lambda_lr = 1e-3
        self.kl_penalty = 1.0
        self.kl_target = 0.5
        self.epoch = 1
        self.rho = self.config.rho
        self.rho_scaler = float(self.config.rho_scaler)
        self.reduction_threshold = self.config.reduction_threshold
        self.lambda_ = torch.zeros(config.layers)
        self.old_violation = torch.full((config.layers,), torch.inf)

        self.lambda_min = torch.zeros(config.layers)
        if config.lambda_min > 0:
            self.lambda_min +=  config.lambda_min

        self.lambda_max = torch.zeros(config.layers) + config.lambda_max
        self.lambda_min = self.lambda_min.to(device)
        self.lambda_max = self.lambda_max.to(device)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = float(self.config.beta)
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def _layer_target(self, layered_kl: torch.Tensor) -> torch.Tensor:
        return layered_kl.new_full(
            layered_kl.shape,
            float(self.kl_target) / max(1, layered_kl.numel()),
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

        residual = expected_kl - self._layer_target(expected_kl)
        constrain_violation = residual.abs()
        if torch.any(
            constrain_violation
            > self.reduction_threshold
            * self.old_violation.to(constrain_violation.device)
        ).item():
            self.rho *= self.rho_scaler
        self.old_violation = constrain_violation
        self.lambda_ = torch.clamp(
            self.lambda_.to(residual.device) + self.rho * residual,
            self.lambda_min,
            self.lambda_max,
        )
        log.info(
            "Outer step: expected KL=%.4f, residual=%.4f, lambda=%s, rho=%.4f, Expected Wasserstein=%.4f, latent mean diff=%.4f, latent logvar diff=%.4f",
            expected_kl.mean().item(),
            residual.mean().item(),
            self.lambda_.cpu().numpy().tolist(),
            self.rho,
            expected_wasserstein.mean().item(),
            latent_mean_diff.mean().item(),
            latent_logvar_diff.mean().item(),
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        _inner=None,
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
        
        lambda_ = self.lambda_.to(residual.device)
        shifted = lambda_ + self.rho * residual
        al_penalty = (torch.clamp(shifted, min=0.0) ** 2 - lambda_**2) / (
            2.0 * self.rho
        )
        loss = rec + al_penalty.sum()
        loss.backward()
        optimizer.step()

        self.last_train_metrics = {
            "posterior_recon": float(rec.detach()),
            "posterior_kl": float(layered_kl.detach().sum()),
            "layered_kl": layered_kl.detach(),
        }

        return float(loss.detach())

    def print_gradients(self, x: torch.Tensor, y: torch.Tensor):
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
        residual = layered_kl - self._layer_target(layered_kl)
        lambda_ = self.lambda_.to(residual.device)
        shifted = lambda_ + self.rho * residual
        al_penalty = (torch.clamp(shifted, min=0.0) ** 2 - lambda_**2) / (
            2.0 * self.rho
        )
        al_penalty = al_penalty.sum()

        grad_info = self.gradient_diagnostics(
            {
                "reconstruction": rec,
                "kl_constraint": al_penalty,
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
        log.info("Cosine reconstruction, kl_constraint:")
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

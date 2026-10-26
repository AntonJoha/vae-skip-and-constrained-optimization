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
    posterior_reduce = "last"
    reverse_posterior_list = False

    def __init__(self, config):

        torch.autograd.set_detect_anomaly(True)
        torch.set_warn_always(True)


        super().__init__(config)
        self.lambda_ = 0.0
        self.lambda_lr = float(getattr(config, "lr_lambda", self.lambda_lr))
        self.rho_lr = float(getattr(config, "rho_lr", self.rho_lr))
        self.kl_penalty = 1.0
        self.kl_target = 0.5
        self.rho = self.config.rho
        self.rho_scaler = float(self.config.rho_scaler)
        self.reduction_threshold = self.config.reduction_threshold
        lambda_min = max(float(config.lambda_min), 1e-3)
        self.lambda_min = torch.full((config.layers,), lambda_min, device=device)
        self.lambda_max = torch.full(
            (config.layers,), float(config.lambda_max), device=device
        )
        self.lambda_ = self.lambda_min.clone()
        self.old_violation = torch.full((config.layers,), torch.inf)

        if config.set_lambda is not None:
            self.lambda_ = torch.zeros(config.layers).to(device) + config.set_lambda
            log.info("Hello")
            for i in range(config.layers):
                self.lambda_[i] = self.lambda_[i]*config.lambda_scaler**(i)
            log.info(str(self.lambda_))



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
        middle_optimizer: torch.optim.Optimizer,
        inner_optimizer,
    ) -> float:
        self.train()
        inner_optimizer.zero_grad()

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
        inner_optimizer.step()

        middle_optimizer.zero_grad()
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
        residual = self._layer_target(layered_kl) - layered_kl
        lambda_ = self.lambda_.to(residual.device)
        shifted = lambda_ + self.rho * residual
        al_penalty = (torch.clamp(shifted, min=0.0) ** 2 - lambda_**2) / (
            2.0 * self.rho
        )
        loss = rec + al_penalty.sum()

        loss.backward()
        middle_optimizer.step()

        dual_lr = middle_optimizer.param_groups[0]["lr"] / 100
        self.lambda_lr = dual_lr
        self.rho_lr = dual_lr
        violation = torch.clamp(residual.detach(), min=0.0)
        self.rho = max(1e-8, self.rho + self.rho_lr * float(violation.mean().item()))
        if self.config.set_lambda is None:
            self.lambda_ = torch.clamp(
                self.lambda_.to(residual.device) + self.lambda_lr * residual.detach(),
                self.lambda_min,
                self.lambda_max,
            )

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
        residual = self._layer_target(layered_kl) - layered_kl
        lambda_ = self.lambda_.to(residual.device)
        shifted = lambda_ + self.rho * residual
        al_penalty = (torch.clamp(shifted, min=0.0) ** 2 - lambda_**2) / (
            2.0 * self.rho
        )
        al_pentaly = al_penalty.sum()

        grad_info = self.gradient_diagnostics(
            {
                "reconstruction": rec,
                "al_pentaly": al_pentaly,
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
            al_pentaly.item(),
        )


        grad_info = self.gradient_diagnostics(
            {
                "full_loss": rec + al_pentaly,
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
            al_pentaly.item(),
        )



        cosine_similarity = self.gradient_cosine_similarity(rec, al_pentaly)

        log.info(
            "  Cosine reconstruction, al_penalty:",
        )
        for group_name, similarity in cosine_similarity.items():
            log.info("    %-20s %.6f", group_name, similarity)

        cosine_similarity = self.gradient_cosine_similarity(prior_kl, al_pentaly)
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
        """
        log.info(
            "  posterior mean=%.6f logvar=%.6f prior mean=%.6f prior logvar=%.6f",
            mean[0][0][0].item(),
            logvar[0][0][0].item(),
            prior_mean[0][0][0].item(),
            prior_logvar[0][0][0].item(),
        )
        """

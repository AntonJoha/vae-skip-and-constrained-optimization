import torch

import logging
from .kl_augmented import Model as AugmentedModel

log = logging.getLogger(__name__)

class Model(AugmentedModel):
    def _posterior_stats(self, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
        y_full = torch.cat([x, y], dim=1)
        posterior = self._encode_state(self.posterior_state, y_full)
        posterior_list: list[torch.Tensor] = []
        for layer in self.posterior_layers:
            posterior = layer(posterior)
            mean, logvar = posterior.chunk(2, dim=-1)
            logvar = self._clamp_logvar(logvar)
            posterior = torch.cat([mean, logvar], dim=-1)
            posterior_list.append(posterior)
            posterior = self._reparametrize(mean, logvar)
        if self._should_reverse_posterior():
            posterior_list.reverse()
        return posterior_list

    def inner_parameters(self):
        return list(self.prior_state.parameters()) + list(
            self.prior_layers.parameters()
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        inner_optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        self.train()

        if inner_optimizer is not None:
            inner_optimizer.zero_grad(set_to_none=True)
            _, _, prior_list, _ = self._latent_pass(x, y, prior=False)
            posterior_list = [t.detach() for t in self._posterior_stats(x, y)]
            prior_match_loss = self._kl_from_stats(prior_list, posterior_list)
            prior_match_loss.backward()
            inner_optimizer.step()

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

        self.last_train_metrics = {
            "posterior_recon": float(rec.detach()),
            "posterior_kl": float(layered_kl.detach().sum()),
            "layered_kl": layered_kl.detach(),
        }

        return float(loss.detach())


    def print_gradients(self, x: torch.Tensor, y: torch.Tensor):


        _, _, prior_list, _ = self._latent_pass(x, y, prior=False)
        posterior_list = [t.detach() for t in self._posterior_stats(x, y)]
        prior_match_loss = self._kl_from_stats(prior_list, posterior_list)


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
                "prior_match_loss": prior_match_loss,
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
        log.info(
            "  posterior mean=%.6f logvar=%.6f prior mean=%.6f prior logvar=%.6f",
            mean[0][0][0].item(),
            logvar[0][0][0].item(),
            prior_mean[0][0][0].item(),
            prior_logvar[0][0][0].item(),
        )



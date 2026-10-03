import logging

import torch

from experiments.util import SeriesConfig

from .kl_base import KLBaseModel

log = logging.getLogger(__name__)

TDLGMConfig = SeriesConfig


class Model(KLBaseModel):
    state_dropout = 0.1
    layer_dropout = 0.1
    logvar_clamp = (-10.0, 2.0)
    combine_gaussians_mode = "product"
    posterior_reduce = "mean"
    reverse_posterior_list = True

    def __init__(self, config):
        super().__init__(config)
        self.last_train_metrics = None
        self.kl_target = float(self.config.beta)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = float(self.config.beta)
        self.kl_target = 0
        if not self.basic:
            self.kl_target = float(self.config.beta) / ((self.epoch + 1) ** 0.5)
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

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

        expected_total_kl = expected_kl.sum()
        residual = expected_total_kl - self.kl_target
        constrain_violation = abs(residual.item())

        if constrain_violation > self.reduction_threshold * self.old_violation:
            self.rho *= self.rho_scaler

        self.old_violation = constrain_violation
        self.lambda_ = min(max(-50, self.lambda_ + self.rho * residual.item()), 50)

        log.info(
            "Outer step: expected KL=%.4f, residual=%.4f, lambda=%.4f, rho=%.4f, Expected Wasserstein=%.4f, latent mean diff=%.4f, latent logvar diff=%.4f",
            expected_kl.mean().item(),
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
        total_kl = layered_kl.sum()
        residual = total_kl - self.kl_target

        kl_constraint = self.lambda_ * residual + 0.5 * self.rho * residual.pow(2)
        loss = rec + kl_constraint

        if self.config.grad_diagnostics:
            grad_info = self._gradient_diagnostics(
                {
                    "reconstruction": rec,
                    "kl_constraint": kl_constraint,
                }
            )

            print("\nGradient norms:")
            for loss_name, groups in grad_info.items():
                print(f"  {loss_name}:")
                for group_name, norm in groups.items():
                    print(f"    {group_name:20s}: {norm:.6e}")
            print("Losses: Reconstruction: ", rec.item(), " kl_cons: ", kl_constraint.item())
            m, l, *_ = self._latent_pass(x, y=None, prior=True)
            print(
                "Logvar posterior: ",
                logvar[0][0][0].item(),
                " Posterior mean: ",
                mean[0][0][0].item(),
                "\nLogvar prior",
                l[0][0][0].item(),
                "Prior mean: ",
                m[0][0][0].item(),
            )

        loss.backward()
        optimizer.step()

        self.last_train_metrics = {
            "posterior_recon": float(rec.detach()),
            "posterior_kl": float(layered_kl.detach().sum()),
            "layered_kl": layered_kl.detach(),
        }

        return float(loss.detach())

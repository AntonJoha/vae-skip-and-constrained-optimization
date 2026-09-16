import logging

import torch
from torch import nn

from experiments.util import SeriesConfig

log = logging.getLogger(__name__)

VRNNConfig = SeriesConfig


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Model(nn.Module):
    def __init__(self, config: VRNNConfig):
        super().__init__()
        self.config = config
        self.prior_encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.posterior_encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.posterior_target_encoder = nn.LSTM(
            input_size=config.output_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.prior_head = _make_mlp(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2 * config.hidden_dim,
        )
        self.posterior_head = _make_mlp(
            input_dim=2 * config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2 * config.hidden_dim,
        )
        self.decoder = _make_mlp(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2 * config.output_dim * config.horizon,
        )
self.nll_loss = nn.GaussianNLLLoss()
        self.nllLoss = self.nll_loss

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def _target(self, y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        target = y.squeeze(-1)
        if mean.shape != target.shape:
            raise ValueError(
                "prediction and target shapes must match after optional "
                "target squeeze of the last dimension: "
                f"{mean.shape} != {target.shape}"
            )
        return target

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _latent_pass(
        self, x: torch.Tensor, y: torch.Tensor | None = None, prior: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) != self.config.input_dim:
            raise ValueError(
                "expected x feature dimension to match config.input_dim: "
                f"{x.size(-1)} != {self.config.input_dim}"
            )

        prior_state, _ = self.prior_encoder(x)
        p_mean, p_logvar = self.prior_head(prior_state[:, -1, :]).chunk(2, dim=-1)

        if prior or y is None:
            q_mean, q_logvar = p_mean, p_logvar
            z = p_mean
        else:
            if y.ndim == 2:
                y = y.unsqueeze(-1)
            if y.ndim != x.ndim:
                raise ValueError(
                    f"expected y and x to have same rank for posterior encoding: "
                    f"{y.ndim} != {x.ndim}"
                )
            context_state, _ = self.posterior_encoder(x)
            target_state, _ = self.posterior_target_encoder(y)
            posterior_state = torch.cat(
                [context_state[:, -1, :], target_state[:, -1, :]], dim=-1
            )
            q_mean, q_logvar = self.posterior_head(posterior_state).chunk(2, dim=-1)
            z = self._reparameterize(q_mean, q_logvar)

        output = self.decoder(z)
        mean, logvar = output.chunk(2, dim=-1)
        pred_mean = self._to_output_shape(mean)
        pred_logvar = self._to_output_shape(logvar.clamp(min=-6.0, max=6.0))

        return (
            pred_mean,
            pred_logvar,
            [torch.cat([p_mean, p_logvar], dim=-1)],
            [torch.cat([q_mean, q_logvar], dim=-1)],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar, *_ = self._latent_pass(x, y=None, prior=True)
        return mean, logvar

    def _compute_losses(
        self,
        y: torch.Tensor,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        prior_list: list[torch.Tensor],
        posterior_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = self._target(y, pred_mean)
        pred_var = pred_logvar.exp()
        pred_mean_flat = pred_mean.reshape(pred_mean.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        pred_var_flat = pred_var.reshape(pred_var.size(0), -1)
        recon_loss = self.nll_loss(pred_mean_flat, target_flat, pred_var_flat)
        kl_loss = self._kl_from_stats(prior_list, posterior_list)
        return recon_loss, kl_loss

    def _kl_from_stats(
        self, prior_list: list[torch.Tensor], posterior_list: list[torch.Tensor]
    ) -> torch.Tensor:
        kl_loss = 0.0
        for prior, posterior in zip(prior_list, posterior_list, strict=False):
            p_mean, p_logvar = prior.chunk(2, dim=-1)
            q_mean, q_logvar = posterior.chunk(2, dim=-1)
            kl = 0.5 * (
                p_logvar
                - q_logvar
                + (torch.exp(q_logvar) + (q_mean - p_mean).pow(2)) / torch.exp(p_logvar)
                - 1
            )
            kl_loss += kl.sum(dim=-1).mean()
        return kl_loss

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        mean, logvar, prior_list, posterior_list = self._latent_pass(x, y, prior=False)
        rec, kl = self._compute_losses(y, mean, logvar, prior_list, posterior_list)
        loss = rec + self.config.beta * kl
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def compute_losses(
        self, x: torch.Tensor, y: torch.Tensor, prior: bool = True
    ) -> tuple[float, float]:
        mean, logvar, prior_list, posterior_list = self._latent_pass(x, y, prior=prior)
        rec, kl = self._compute_losses(y, mean, logvar, prior_list, posterior_list)
        return float(rec), float(kl)

    @torch.no_grad()
    def get_layered_kl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, prior_list, posterior_list = self._latent_pass(x, y, prior=False)
        kl = self._kl_from_stats(prior_list, posterior_list)
        return torch.tensor([float(kl)], device=x.device)

from __future__ import annotations

import torch
from torch import nn

from experiments.util import SeriesConfig

TDLGMConfig = SeriesConfig


def _flatten_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim > 2:
        return x.flatten(start_dim=1)
    return x


def _normalize_target(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y.unsqueeze(-1)
    if y.ndim > 2:
        return y.flatten(start_dim=1)
    return y


def _diag_kl_divergence(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    posterior_var = posterior_logvar.exp()
    prior_var = prior_logvar.exp()
    return 0.5 * (
        (posterior_var / prior_var)
        + ((posterior_mean - prior_mean).pow(2) / prior_var)
        - 1.0
        + prior_logvar
        - posterior_logvar
    ).sum(dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class LadderLatentBlock(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.bottom_up = nn.Linear(hidden_dim, 2 * latent_dim)
        self.top_down = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(
        self,
        bottom_up_state: torch.Tensor,
        top_down_state: torch.Tensor | None,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bu_mean, bu_logvar = self.bottom_up(bottom_up_state).chunk(2, dim=-1)

        if top_down_state is None:
            td_mean = torch.zeros_like(bu_mean)
            td_logvar = torch.zeros_like(bu_logvar)
        else:
            td_mean, td_logvar = self.top_down(top_down_state).chunk(2, dim=-1)

        bu_precision = bu_logvar.neg().exp() * alpha
        td_precision = td_logvar.neg().exp()
        posterior_var = (bu_precision + td_precision).reciprocal().clamp_min(1e-6)
        posterior_mean = posterior_var * (
            bu_precision * bu_mean + td_precision * td_mean
        )
        posterior_logvar = posterior_var.log()
        return posterior_mean, posterior_logvar, td_mean, td_logvar


class Model(nn.Module):
    def __init__(self, config: SeriesConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.num_latent_layers = max(1, config.tdlgm_layers)
        self.beta = config.beta
        self.alpha = max(1.0, float(config.alpha))

        self.nllLoss = nn.GaussianNLLLoss()

        self.input_encoder = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        encoder_depth = max(config.layers, self.num_latent_layers)
        self.encoder_blocks = nn.ModuleList(
            ResidualBlock(self.hidden_dim) for _ in range(encoder_depth)
        )
        self.latent_blocks = nn.ModuleList(
            LadderLatentBlock(self.hidden_dim, self.latent_dim)
            for _ in range(self.num_latent_layers)
        )
        self.latent_to_hidden = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            )
            for _ in range(self.num_latent_layers)
        )
        self.decoder_blocks = nn.ModuleList(
            ResidualBlock(self.hidden_dim) for _ in range(self.num_latent_layers)
        )
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2 * config.output_dim),
        )

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.input_encoder(_flatten_sequence(x))
        states = [h]
        for block in self.encoder_blocks:
            h = block(h)
            states.append(h)
        return states

    def _sample_latents(
        self, x: torch.Tensor, sample: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.training if sample is None else sample
        encoder_states = self._encode(x)
        top_down_state: torch.Tensor | None = None
        kl_terms: list[torch.Tensor] = []
        latent_samples: list[torch.Tensor] = []

        for idx, latent_block in enumerate(reversed(self.latent_blocks)):
            state_index = min(idx, len(encoder_states) - 1)
            bottom_up_state = encoder_states[-(state_index + 1)]
            posterior_mean, posterior_logvar, prior_mean, prior_logvar = latent_block(
                bottom_up_state,
                top_down_state,
                self.alpha,
            )
            if sample:
                z = posterior_mean + torch.randn_like(posterior_mean) * (
                    0.5 * posterior_logvar
                ).exp()
            else:
                z = posterior_mean
            kl_terms.append(
                _diag_kl_divergence(
                    posterior_mean,
                    posterior_logvar,
                    prior_mean,
                    prior_logvar,
                )
            )
            latent_samples.append(z)

            top_down_state = self.decoder_blocks[idx](self.latent_to_hidden[idx](z))

        latent_samples.reverse()
        kl_terms.reverse()

        assert top_down_state is not None
        latent_context = top_down_state
        return latent_context, torch.stack(kl_terms, dim=0).sum(dim=0), torch.stack(
            latent_samples, dim=0
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_context, _, _ = self._sample_latents(x, sample=self.training)
        mean, logvar = self.output_head(latent_context).chunk(2, dim=-1)
        return mean, logvar

    def _compute_losses(
        self, y: torch.Tensor, pred_mean: torch.Tensor, pred_logvar: torch.Tensor, kl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = _normalize_target(y)
        recon_loss = self.nllLoss(
            pred_mean,
            target,
            pred_logvar.clamp(min=-12.0, max=8.0).exp().clamp_min(1e-6),
        )
        kl_loss = kl.mean()
        return recon_loss, kl_loss

    @torch.no_grad()
    def compute_losses(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        latent_context, kl, _ = self._sample_latents(x, sample=self.training)
        mean, logvar = self.output_head(latent_context).chunk(2, dim=-1)
        recon, kl_loss = self._compute_losses(y, mean, logvar, kl)
        return float(recon), float(kl_loss)

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        latent_context, kl, _ = self._sample_latents(x, sample=True)
        mean, logvar = self.output_head(latent_context).chunk(2, dim=-1)
        recon_loss, kl_loss = self._compute_losses(y, mean, logvar, kl)
        loss = recon_loss + self.beta * kl_loss
        loss.backward()
        optimizer.step()
        return float(loss)

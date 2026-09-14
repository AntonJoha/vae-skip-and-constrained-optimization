import logging
import random

import torch
from torch import nn

from experiments.util import SeriesConfig

log = logging.getLogger(__name__)

TDLGMConfig = SeriesConfig


def _resolve_num_heads(hidden_dim: int) -> int:
    for candidate in (8, 4, 2, 1):
        if hidden_dim % candidate == 0 and hidden_dim // candidate >= 16:
            return candidate
    return 1


def _kl_divergence(
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
) -> torch.Tensor:
    prior_var = prior_logvar.exp()
    posterior_var = posterior_logvar.exp()
    return 0.5 * (
        prior_logvar
        - posterior_logvar
        + (posterior_var + (posterior_mean - prior_mean).pow(2)) / prior_var
        - 1
    ).sum(dim=-1)


class SequenceAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, seq_len: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=_resolve_num_heads(hidden_dim),
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.size(1) > self.position_embedding.size(1):
            raise ValueError(
                "sequence length exceeds configured maximum: "
                f"{x.size(1)} > {self.position_embedding.size(1)}"
            )
        h = self.input_proj(x) + self.position_embedding[:, : x.size(1), :]
        return self.norm(self.encoder(h))


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.posterior_state = SequenceAttentionEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=2,
            seq_len=config.seq_len + config.horizon,
        )
        self.prior_state = SequenceAttentionEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=2,
            seq_len=config.seq_len,
        )
        self.to_output = nn.Linear(
            config.hidden_dim, 2 * config.output_dim * config.horizon
        )

        self.downwards_list = self.make_layers(
            config.hidden_dim, config.hidden_dim, config.layers
        )
        self.upwards_list = self.make_layers(
            config.hidden_dim, config.hidden_dim, config.layers
        )

        self.skip_connection = config.skip_connection
        if self.skip_connection:
            log.info("Using skip connections with %d layers", config.layers)
            self.skip_weights = self.make_skips(config.layers)

        self.nllLoss = nn.GaussianNLLLoss()
        self.lambda_ = 1.0
        self.lambda_lr = 1.0
        self.kl_target = 0.5

    def make_skips(self, num_layers):
        return nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)]
        )

    def make_layers(self, hidden_dim, output_dim, num_layers):
        return nn.ModuleList(
            [_make_mlp(hidden_dim, output_dim, 2 * output_dim) for _ in range(num_layers)]
        )

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def _reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def _multiply_gaussians(
        self,
        mean1: torch.Tensor,
        logvar1: torch.Tensor,
        mean2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_tau1, log_tau2 = -logvar1, -logvar2
        stacked_log_taus = torch.stack([log_tau1, log_tau2], dim=-1)
        combined_logvar = -torch.logsumexp(stacked_log_taus, dim=-1)

        max_log_tau = stacked_log_taus.max(dim=-1).values
        exp_diff1 = torch.exp(log_tau1 - max_log_tau)
        exp_diff2 = torch.exp(log_tau2 - max_log_tau)
        combined_mean = (mean1 * exp_diff1 + mean2 * exp_diff2) / (exp_diff1 + exp_diff2)
        return combined_mean, combined_logvar

    def _latent_pass(self, x, y=None, prior=True) -> torch.Tensor:
        posterior_list = []
        combined_posterior_list = []

        if y is not None:
            posterior = self.posterior_state(torch.cat([x, y], dim=1))[:, -1, :]
            for layer in self.downwards_list:
                posterior = layer(posterior)
                posterior_list.append(posterior)
                mean, logvar = posterior.chunk(2, dim=-1)
                posterior = self._reparametrize(mean, logvar)
            posterior_list.reverse()

        prior_state = self.prior_state(x)[:, -1, :]
        prior_list = []
        for i, layer in enumerate(self.upwards_list):
            residual = prior_state
            prior_state = layer(prior_state)
            prior_list.append(prior_state)

            if prior:
                mean, logvar = prior_state.chunk(2, dim=-1)
                prior_state = self._reparametrize(mean, logvar)
            else:
                posterior = posterior_list[i]
                q_mean, q_logvar = posterior.chunk(2, dim=-1)
                p_mean, p_logvar = prior_state.chunk(2, dim=-1)
                mean, logvar = self._multiply_gaussians(
                    q_mean, q_logvar, p_mean, p_logvar
                )
                combined_posterior_list.append(torch.cat([mean, logvar], dim=-1))
                prior_state = self._reparametrize(mean, logvar)

            if self.skip_connection:
                prior_state = prior_state + self.skip_weights[i] * residual

        output = self.to_output(prior_state)
        mean, logvar = output.chunk(2, dim=-1)
        return (
            self._to_output_shape(mean),
            self._to_output_shape(logvar),
            prior_list,
            combined_posterior_list,
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = 1 / (self.epoch**self.config.beta) if self.epoch > 0 else 1.0
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar, *_ = self._latent_pass(x, y=None, prior=True)
        return mean, logvar

    def _compute_losses(
        self,
        y,
        pred_mean,
        pred_logvar,
        prior_list=None,
        combined_posterior_list=None,
    ):
        recon_loss = self.nllLoss(pred_mean, y.squeeze(-1), pred_logvar.exp())
        kl_loss = 0.0
        if prior_list is not None and combined_posterior_list is not None:
            for prior, posterior in zip(prior_list, combined_posterior_list):
                p_mean, p_logvar = prior.chunk(2, dim=-1)
                q_mean, q_logvar = posterior.chunk(2, dim=-1)
                kl_loss += _kl_divergence(p_mean, p_logvar, q_mean, q_logvar).mean()
        return recon_loss, kl_loss

    def _layered_kl(self, prior_list, combined_posterior_list):
        if not prior_list:
            return torch.empty(0, device=self.to_output.weight.device)
        return torch.stack(
            [
                _kl_divergence(
                    *prior.chunk(2, dim=-1),
                    *posterior.chunk(2, dim=-1),
                ).mean()
                for prior, posterior in zip(prior_list, combined_posterior_list)
            ]
        )

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)

        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False
        )
        rec, kl = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=prior_list,
            combined_posterior_list=combined_posterior_list,
        )
        loss = rec + self.lambda_ * torch.relu(kl - self.kl_target)

        loss.backward()
        optimizer.step()

        if random.random() < 0.01:
            with torch.no_grad():
                self.lambda_ = max(
                    0.0,
                    self.lambda_ + self.lambda_lr * float((kl - self.kl_target).item()),
                )

        return float(loss.detach())

    @torch.no_grad()
    def get_layered_kl(self, x, y):
        _, _, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False
        )
        return self._layered_kl(prior_list, combined_posterior_list).detach()

    @torch.no_grad()
    def compute_losses(self, x: torch.Tensor, y: torch.Tensor, prior: bool = True):
        pred_mean, pred_logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=prior
        )
        rec, kl = self._compute_losses(
            y,
            pred_mean,
            pred_logvar,
            prior_list=prior_list,
            combined_posterior_list=combined_posterior_list,
        )
        return float(rec), float(kl)

from __future__ import annotations

import logging

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


class SequenceRNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, seq_len: int):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        output, _ = self.rnn(x)
        return output


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
        h = self.encoder(h)
        return self.norm(h)


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.use_attention = bool(getattr(config, "attention", False))
        self.use_mean_pooling = bool(getattr(config, "use_old", False))
        self.skip_connection = bool(config.skip_connection)

        encoder_cls = SequenceAttentionEncoder if self.use_attention else SequenceRNNEncoder
        self.posterior_state = encoder_cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=config.layers,
            seq_len=config.seq_len + config.horizon,
        )
        self.prior_state = encoder_cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=config.layers,
            seq_len=config.seq_len,
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
            nn.Linear(self.hidden_dim, 2 * config.output_dim * config.horizon),
        )

        self.skip_weights = (
            nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.num_latent_layers)])
            if self.skip_connection
            else None
        )

        self.nllLoss = nn.GaussianNLLLoss()
        self.epoch = 0
        self.kl_target = self.beta

    def _pool_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=1) if self.use_mean_pooling else hidden[:, -1, :]

    def _encode(self, x: torch.Tensor, posterior: bool) -> list[torch.Tensor]:
        encoder = self.posterior_state if posterior else self.prior_state
        h = self._pool_sequence(encoder(x))
        states = [h]
        for block in self.encoder_blocks:
            h = block(h)
            states.append(h)
        return states

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def _reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def _latent_pass(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        prior: bool = True,
        sample: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        sample = self.training if sample is None else sample
        use_posterior = y is not None and not prior
        sequence = torch.cat([x, y], dim=1) if use_posterior else x
        encoder_states = self._encode(sequence, posterior=use_posterior)

        top_down_state: torch.Tensor | None = None
        prior_list: list[torch.Tensor] = []
        combined_posterior_list: list[torch.Tensor] = []

        for idx, latent_block in enumerate(reversed(self.latent_blocks)):
            state_index = min(idx, len(encoder_states) - 1)
            bottom_up_state = encoder_states[-(state_index + 1)]
            posterior_mean, posterior_logvar, prior_mean, prior_logvar = latent_block(
                bottom_up_state,
                top_down_state,
                self.alpha,
            )
            prior_list.append(torch.cat([prior_mean, prior_logvar], dim=-1))

            if use_posterior:
                if sample:
                    z = self._reparametrize(posterior_mean, posterior_logvar)
                else:
                    z = posterior_mean
                combined_posterior_list.append(
                    torch.cat([posterior_mean, posterior_logvar], dim=-1)
                )
                kl_term = None
            else:
                if sample:
                    z = self._reparametrize(prior_mean, prior_logvar)
                else:
                    z = prior_mean

            top_down_state = self.decoder_blocks[idx](self.latent_to_hidden[idx](z))
            if self.skip_connection:
                top_down_state = top_down_state + self.skip_weights[idx] * bottom_up_state

        assert top_down_state is not None
        output = self.output_head(top_down_state)
        mean, logvar = output.chunk(2, dim=-1)
        pred_mean = self._to_output_shape(mean)
        pred_logvar = torch.clamp(self._to_output_shape(logvar), -12.0, 8.0)
        return pred_mean, pred_logvar, prior_list, combined_posterior_list

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar, *_ = self._latent_pass(x, prior=True)
        return mean, logvar

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = self.beta
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def _compute_losses(
        self,
        y: torch.Tensor,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        prior_list: list[torch.Tensor] | None = None,
        combined_posterior_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = _normalize_target(y)
        recon_loss = self.nllLoss(pred_mean, target, pred_logvar.exp().clamp_min(1e-6))
        kl_loss = torch.tensor(0.0, device=pred_mean.device)
        if prior_list is not None and combined_posterior_list is not None:
            for prior, posterior in zip(prior_list, combined_posterior_list):
                p_mean, p_logvar = prior.chunk(2, dim=-1)
                q_mean, q_logvar = posterior.chunk(2, dim=-1)
                kl_loss = kl_loss + _diag_kl_divergence(
                    q_mean,
                    q_logvar,
                    p_mean,
                    p_logvar,
                ).mean()
        return recon_loss, kl_loss

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False, sample=True
        )
        recon_loss, kl_loss = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=prior_list,
            combined_posterior_list=combined_posterior_list,
        )
        loss = recon_loss + self.beta * kl_loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def compute_losses(self, x: torch.Tensor, y: torch.Tensor, prior: bool = True):
        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(
            x, None if prior else y, prior=prior, sample=False
        )
        recon_loss, kl_loss = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=prior_list if not prior else None,
            combined_posterior_list=combined_posterior_list if not prior else None,
        )
        return float(recon_loss), float(kl_loss)

    @torch.no_grad()
    def get_layered_kl(self, x: torch.Tensor, y: torch.Tensor):
        _, _, prior_list, combined_posterior_list = self._latent_pass(
            x, y, prior=False, sample=False
        )
        kl_losses = []
        for prior, posterior in zip(prior_list, combined_posterior_list):
            p_mean, p_logvar = prior.chunk(2, dim=-1)
            q_mean, q_logvar = posterior.chunk(2, dim=-1)
            kl_losses.append(
                _diag_kl_divergence(q_mean, q_logvar, p_mean, p_logvar).mean()
            )
        return torch.stack(kl_losses) if kl_losses else torch.empty(0, device=x.device)


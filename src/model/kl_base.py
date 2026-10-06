import logging
import numpy as np

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


class SequenceRNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        seq_len: int,  # unused, kept for compatibility
    ):
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
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=_resolve_num_heads(hidden_dim),
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
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


def _make_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
    ]
    if dropout:
        layers.append(nn.Dropout(p=dropout))
    layers.extend(
        [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
    )
    if dropout:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class KLBaseModel(nn.Module):
    state_dropout = 0.0
    layer_dropout = 0.0
    logvar_clamp: tuple[float, float] | None = None
    combine_gaussians_mode = "product"
    posterior_reduce = "last"
    prior_reduce = "last"
    reverse_posterior_list = False

    def __init__(self, config: TDLGMConfig):
        super().__init__()
        self.config = config
        self.basic = getattr(config, "vae_baseline", False)
        self.epoch = 0
        self.model = _make_mlp(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2 * config.output_dim * config.horizon,
            dropout=self.layer_dropout,
        )
        self.posterior_state = SequenceAttentionEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=2,
            seq_len=config.seq_len + config.horizon,
            dropout=self.state_dropout,
        )
        self.prior_state = SequenceAttentionEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=2,
            seq_len=config.seq_len,
            dropout=self.state_dropout,
        )
        self.to_output = nn.Linear(
            config.hidden_dim, 2 * config.output_dim * config.horizon
        )
        self.posterior_layers = self.make_layers(
            config.hidden_dim, config.hidden_dim, config.layers
        )
        self.prior_layers = self.make_layers(
            config.hidden_dim, config.hidden_dim, config.layers
        )
        self.skip_connection = config.skip_connection
        if self.skip_connection:
            log.info("Using skip connections with %d layers", config.layers)
            self.skip_weights = self.make_skips(config.layers)
        self.nllLoss = nn.GaussianNLLLoss()
        self.last_train_metrics = None
        self.lambda_ = 0.0
        self.lambda_lr = float(getattr(config, "lr_lambda", 1e-3))
        self.kl_target = float(config.beta)
        self.kl_penalty = 1.0
        self.rho = float(getattr(config, "rho", 1.0))
        self.rho_scaler = float(getattr(config, "rho_scaler", 1.0))
        self.reduction_threshold = float(getattr(config, "reduction_threshold", 1.0))
        self.old_violation = torch.inf

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def make_skips(self, num_layers: int) -> nn.ParameterList:
        return nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)]
        )

    def make_layers(
        self, hidden_dim: int, output_dim: int, num_layers: int
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                _make_mlp(
                    hidden_dim,
                    output_dim,
                    2 * output_dim,
                    dropout=self.layer_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def _reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        if self.logvar_clamp is None:
            return logvar
        lower, upper = self.logvar_clamp
        return torch.clamp(logvar, lower, upper)

    def _combine_gaussians(
        self,
        mean1: torch.Tensor,
        logvar1: torch.Tensor,
        mean2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.combine_gaussian:
            return mean1, logvar1

        log_tau1, log_tau2 = -logvar1, -logvar2
        stacked_log_taus = torch.stack([log_tau1, log_tau2], dim=-1)
        log_tau_comb = torch.logsumexp(stacked_log_taus, dim=-1)
        combined_logvar = -log_tau_comb
        max_log_tau = torch.max(stacked_log_taus, dim=-1).values
        exp_diff1 = torch.exp(log_tau1 - max_log_tau)
        exp_diff2 = torch.exp(log_tau2 - max_log_tau)
        combined_mean = (mean1 * exp_diff1 + mean2 * exp_diff2) / (
            exp_diff1 + exp_diff2
        )
        return combined_mean, combined_logvar

    def _reduce_encoder_output(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.posterior_reduce == "mean":
            return encoded.mean(dim=1)
        return encoded[:, -1, :]

    def _encode_state(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        encoded = encoder(x)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        return self._reduce_encoder_output(encoded)

    def _should_reverse_posterior(self) -> bool:
        return self.reverse_posterior_list and getattr(self.config, "reverse", False)

    def _latent_pass(
        self, x: torch.Tensor, y: torch.Tensor | None = None, prior: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        posterior_list: list[torch.Tensor] = []
        combined_posterior_list: list[torch.Tensor] = []

        if y is not None:
            y_full = torch.cat([x, y], dim=1)
            posterior = self._encode_state(self.posterior_state, y_full)
            for layer in self.posterior_layers:
                posterior = layer(posterior)
                mean, logvar = posterior.chunk(2, dim=-1)
                logvar = self._clamp_logvar(logvar)
                posterior = torch.cat([mean, logvar], dim=-1)
                posterior_list.append(posterior)
                posterior = self._reparametrize(mean, logvar)
            if self._should_reverse_posterior():
                posterior_list.reverse()

        prior_state = self._encode_state(self.prior_state, x)
        prior_list: list[torch.Tensor] = []
        for i, layer in enumerate(self.prior_layers):
            old_state = prior_state
            prior_state = layer(prior_state)
            prior_list.append(prior_state)

            if prior:
                mean, logvar = prior_state.chunk(2, dim=-1)
                logvar = self._clamp_logvar(logvar)
                prior_state = self._reparametrize(mean, logvar)
            else:
                posterior = posterior_list[i]
                q_mean, q_logvar = posterior.chunk(2, dim=-1)
                p_mean, p_logvar = prior_state.chunk(2, dim=-1)
                q_logvar = self._clamp_logvar(q_logvar)
                p_logvar = self._clamp_logvar(p_logvar)
                mean, logvar = self._combine_gaussians(
                    q_mean, q_logvar, p_mean, p_logvar
                )
                logvar = self._clamp_logvar(logvar)
                combined_posterior_list.append(torch.cat([mean, logvar], dim=-1))
                prior_state = self._reparametrize(mean, logvar)

            if self.skip_connection:
                prior_state = prior_state + self.skip_weights[i] * old_state

        output = self.to_output(prior_state)
        mean, logvar = output.chunk(2, dim=-1)
        logvar = self._clamp_logvar(logvar)
        pred_mean = self._to_output_shape(mean)
        pred_logvar = self._to_output_shape(logvar)
        return pred_mean, pred_logvar, prior_list, combined_posterior_list

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar, *_ = self._latent_pass(x, y=None, prior=True)
        return mean, logvar

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

    def _layered_kl(
        self,
        prior_list: list[torch.Tensor],
        combined_posterior_list: list[torch.Tensor],
    ) -> torch.Tensor:
        kl_losses = []
        for prior, posterior in zip(prior_list, combined_posterior_list, strict=False):
            p_mean, p_logvar = prior.chunk(2, dim=-1)
            q_mean, q_logvar = posterior.chunk(2, dim=-1)
            kl = 0.5 * (
                p_logvar
                - q_logvar
                + (torch.exp(q_logvar) + (q_mean - p_mean).pow(2)) / torch.exp(p_logvar)
                - 1
            )
            kl_losses.append(kl.sum(dim=-1).mean())
        return torch.stack(kl_losses)

    def _compute_losses(
        self,
        y: torch.Tensor,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        prior_list: list[torch.Tensor] | None = None,
        combined_posterior_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon_loss = self.nllLoss(pred_mean, y.squeeze(-1), pred_logvar.exp())
        kl_loss = 0.0
        if prior_list is not None and combined_posterior_list is not None:
            kl_loss = self._kl_from_stats(prior_list, combined_posterior_list)
        return recon_loss, kl_loss

    @torch.no_grad()
    def compute_losses(
        self, x: torch.Tensor, y: torch.Tensor, prior: bool = True
    ) -> tuple[float, float]:
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

    @torch.no_grad()
    def get_layered_kl(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, prior_list, combined_posterior_list = self._latent_pass(x, y, prior=False)
        return self._layered_kl(prior_list, combined_posterior_list).detach()

    @torch.no_grad()
    def get_layered_wasserstein(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, prior_list, combined_posterior_list = self._latent_pass(x, y, prior=False)
        wasserstein_losses = []
        for prior, posterior in zip(prior_list, combined_posterior_list, strict=False):
            p_mean, p_logvar = prior.chunk(2, dim=-1)
            q_mean, q_logvar = posterior.chunk(2, dim=-1)
            wasserstein = (
                (
                    (p_mean - q_mean).pow(2)
                    + (
                        torch.sqrt(torch.exp(p_logvar))
                        - torch.sqrt(torch.exp(q_logvar))
                    ).pow(2)
                )
                .sum(dim=-1)
                .mean()
            )
            wasserstein_losses.append(wasserstein.item())
        return torch.tensor(wasserstein_losses, device=x.device)

    @torch.no_grad()
    def get_mean_logvar_diff(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, prior_list, combined_posterior_list = self._latent_pass(x, y, prior=False)
        mean_diffs = []
        logvar_diffs = []
        for prior, posterior in zip(prior_list, combined_posterior_list, strict=False):
            p_mean, p_logvar = prior.chunk(2, dim=-1)
            q_mean, q_logvar = posterior.chunk(2, dim=-1)
            mean_diff = (p_mean - q_mean).pow(2).sum(dim=-1).mean()
            logvar_diff = (
                (torch.sqrt(torch.exp(p_logvar)) - torch.sqrt(torch.exp(q_logvar)))
                .pow(2)
                .sum(dim=-1)
                .mean()
            )
            mean_diffs.append(mean_diff.item())
            logvar_diffs.append(logvar_diff.item())
        return torch.tensor(mean_diffs, device=x.device), torch.tensor(
            logvar_diffs, device=x.device
        )

    def _parameter_groups(self):
        groups = {
            "posterior_encoder": list(self.posterior_state.parameters()),
            "posterior_layers": list(self.posterior_layers.parameters()),
            "prior_encoder": list(self.prior_state.parameters()),
            "prior_layers": list(self.prior_layers.parameters()),
            "output": list(self.to_output.parameters()),
        }
        if self.skip_connection:
            groups["skip_weights"] = list(self.skip_weights.parameters())
        return groups

    def _grad_norm(self, loss, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        num_weights = sum(p.numel() for p in parameters)
        if num_weights == 0:
            return 0.0
        grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = 0.0
        for grad in grads:
            if grad is not None:
                squared_norm += grad.detach().pow(2).sum()
        squared_norm = np.float64(squared_norm)
        return (np.sqrt(squared_norm) / num_weights**0.5).item()

    def inner_parameters(self):
        return list(self.posterior_state.parameters()) + list(
            self.posterior_layers.parameters()
        )

    def middle_parameters(self):
        return (
            list(self.prior_state.parameters())
            + list(self.prior_layers.parameters())
            + list(self.to_output.parameters())
        )

    def _gradient_diagnostics(self, losses):
        groups = self._parameter_groups()
        result = {}
        for loss_name, loss in losses.items():
            result[loss_name] = {}
            for group_name, parameters in groups.items():
                result[loss_name][group_name] = self._grad_norm(loss, parameters)
        return result

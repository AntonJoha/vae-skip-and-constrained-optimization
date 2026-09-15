import logging

import torch
from torch import nn

from experiments.util import SeriesConfig

print("LOADING THIS")
#torch.autograd.set_detect_anomaly(True)
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

        #self.norm = nn.LayerNorm(hidden_dim)

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


class SequenceAttentionEncoderCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, seq_len):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, seq_len, hidden_dim)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=_resolve_num_heads(hidden_dim),
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=layers,
            enable_nested_tensor=False,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        h = self.input_proj(x)
        h = h + self.position_embedding[:, : h.size(1)]

        h = self.encoder(h)

        # [B,T,H] -> [B,H,T]
        h = h.transpose(1, 2)

        h = self.conv(h)

        # [B,H,T] -> [B,T,H]
        h = h.transpose(1, 2)

        return self.norm(h)


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
        self.model = _make_mlp(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=2 * config.output_dim * config.horizon,
        )

        self.posterior_state = SequenceRNNEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=2,
            seq_len=config.seq_len + config.horizon,
        )
        self.prior_state = SequenceAttentionEncoderCNN(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            layers=2,
            seq_len=config.seq_len,
            )

        self.to_output = nn.Linear(config.hidden_dim, 2 * config.output_dim*config.horizon)


        self.to_output = nn.Linear(config.hidden_dim, 2 * config.output_dim*config.horizon)

        self.downwards_list = self.make_layers(config.hidden_dim, config.hidden_dim, config.layers)
        self.upwards_list = self.make_layers(config.hidden_dim, config.hidden_dim, config.layers)

        self.skip_connection = config.skip_connection
        if self.skip_connection:
            log.info("Using skip connections with %d layers", config.layers)
            self.skip_weights = self.make_skips(config.layers)



        self.nllLoss = nn.GaussianNLLLoss()
        self.config = config


        self.lambda_ = 1.0
        self.lambda_lr = 1e-3
        self.kl_target = self.config.beta 

    def make_skips(self, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Parameter(torch.tensor(1.0)))
        return nn.ParameterList(layers)
    
    def make_layers(self, hidden_dim, output_dim, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(_make_mlp(hidden_dim, output_dim, 2 * output_dim))
        return nn.ModuleList(layers)

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:

        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def _reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        #return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _multiply_gaussians(self, mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor):
        #return mean1, logvar1
        
        
        # https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
        # Implementation using the log-precision (log-tau) trick for numerical stability
        log_tau1, log_tau2 = -logvar1, -logvar2

        # log(tau_comb) = log(exp(log_tau1) + exp(log_tau2)) using logsumexp for stability
        stacked_log_taus = torch.stack([log_tau1, log_tau2], dim=-1)
        log_tau_comb = torch.logsumexp(stacked_log_taus, dim=-1)
        combined_logvar = -log_tau_comb

        # Stabilized weighted average for the mean: mu_comb = (mu1*tau1 + mu2*tau2) / (tau1+tau2)
        max_log_tau = torch.max(stacked_log_taus, dim=-1).values
        exp_diff1 = torch.exp(log_tau1 - max_log_tau)
        exp_diff2 = torch.exp(log_tau2 - max_log_tau)
        combined_mean = (mean1 * exp_diff1 + mean2 * exp_diff2) / (exp_diff1 + exp_diff2)

        return combined_mean, combined_logvar

    def _latent_pass(self, x, y=None, prior=True):
    
        posterior_list = []
        combined_posterior_list = []
    
        if y is not None:
    
            y_full = torch.cat([x, y], dim=1)
            posterior = self.posterior_state(y_full).mean(dim=1)
    
            for layer in self.upwards_list:
                posterior = layer(posterior)
                posterior_list.append(posterior)
    
                mean, logvar = posterior.chunk(2, dim=-1)
                logvar = torch.clamp(logvar, -6.0, 2.0)
    
                posterior = self._reparametrize(mean, logvar)
            posterior_list.reverse()

    
        prior_state = self.prior_state(x).mean(dim=1)
    
        prior_list = []
    
        for i, layer in enumerate(self.downwards_list):
    
            old_state = prior_state
    
            prior_state = layer(prior_state)
            prior_list.append(prior_state)
    
            if prior:
    
                mean, logvar = prior_state.chunk(2, dim=-1)
                logvar = torch.clamp(logvar, -6.0, 2.0)
    
                prior_state = self._reparametrize(mean, logvar)
    
            else:
    
                posterior = posterior_list[i]
    
                q_mean, q_logvar = posterior.chunk(2, dim=-1)
                p_mean, p_logvar = prior_state.chunk(2, dim=-1)
    
                q_logvar = torch.clamp(q_logvar, -6.0, 2.0)
                p_logvar = torch.clamp(p_logvar, -6.0, 2.0)
    
                mean, logvar = self._multiply_gaussians(
                    q_mean, q_logvar,
                    p_mean, p_logvar,
                )
    
                logvar = torch.clamp(logvar, -6.0, 2.0)
    
                combined_posterior_list.append(
                    torch.cat([mean, logvar], dim=-1)
                )
    
                prior_state = self._reparametrize(mean, logvar)
    
            if self.skip_connection:
                prior_state = prior_state + self.skip_weights[i] * old_state
    
        output = self.to_output(prior_state)
    
        mean, logvar = output.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -6.0, 2.0)
    
        pred_mean = self._to_output_shape(mean)
        pred_logvar = self._to_output_shape(logvar)
    
        return pred_mean, pred_logvar, prior_list, combined_posterior_list





    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar, *_ = self._latent_pass(x, y=None, prior=True)

        return mean, logvar


    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.kl_target = 1/(self.epoch**self.config.beta) if self.epoch > 0 else 1.0 
        log.info("Epoch %d: KL target set to %.4f", epoch, self.kl_target)

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        mean, logvar, prior_list, combined_posterior_list = self._latent_pass(x, y, prior=False)

        rec, kl = self._compute_losses(
            y,
            mean,
            logvar,
            prior_list=prior_list,
            combined_posterior_list=combined_posterior_list,
        )
        layered_kl = self._layered_kl(prior_list, combined_posterior_list)
        loss = rec + (self.kl_target - layered_kl).pow(2).mean()

        loss.backward()
        optimizer.step()
        return float(loss.detach())


    def _compute_losses(self, y, pred_mean, pred_logvar, prior_list=None, combined_posterior_list=None):
        # Reconstruction loss
        recon_loss = self.nllLoss(pred_mean, y.squeeze(-1), pred_logvar.exp())
        kl_loss = 0.0
        if prior_list is not None and combined_posterior_list is not None:
            for prior, posterior in zip(prior_list, combined_posterior_list):

                p_mean, p_logvar = prior.chunk(2, dim=-1)
                q_mean, q_logvar = posterior.chunk(2, dim=-1)

                kl = 0.5 * (
                    p_logvar - q_logvar
                    + (torch.exp(q_logvar) + (q_mean - p_mean).pow(2))
                      / torch.exp(p_logvar)
                    - 1
                )

                kl_loss += kl.sum(dim=-1).mean()

        return recon_loss, kl_loss

    @torch.no_grad()
    def compute_losses(self, x: torch.Tensor, y: torch.Tensor, prior: bool = True):
        (
            pred_mean,
            pred_logvar,
            prior_list,
            combined_posterior_list,
        ) = self._latent_pass(x, y, prior=prior)
        rec, kl = self._compute_losses(
            y,
            pred_mean,
            pred_logvar,
            prior_list=prior_list,
            combined_posterior_list=combined_posterior_list
        )
        return float(rec), float(kl)
    
    @torch.no_grad()
    def get_layered_kl(self, x, y):
        _, _, prior_list, combined_posterior_list = self._latent_pass(x, y, prior=False)
        return self._layered_kl(prior_list, combined_posterior_list).detach()




    def _layered_kl(self, prior_list, combined_posterior_list):
        kl_losses = []
        for prior, posterior in zip(prior_list, combined_posterior_list):
            p_mean, p_logvar = prior.chunk(2, dim=-1)
            q_mean, q_logvar = posterior.chunk(2, dim=-1)

            kl = 0.5 * (
                p_logvar - q_logvar
                + (torch.exp(q_logvar) + (q_mean - p_mean).pow(2))
                  / torch.exp(p_logvar)
                - 1
            )
            kl_losses.append(kl.sum(dim=-1).mean())



        return torch.stack(kl_losses)

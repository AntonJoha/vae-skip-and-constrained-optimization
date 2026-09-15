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
        self.config = config
        self.lstm = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )

        self.prior_state = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
            )

        self.posterior_state = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
            )

        self.module = _make_mlp(config.hidden_dim, config.hidden_dim, config.hidden_dim*2)

        self.linear = nn.Linear(
            config.hidden_dim, config.output_dim * 2 * config.horizon
        )
        self.nllLoss= nn.GaussianNLLLoss()
        self.config = config


    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _to_output_shape(self, x):
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._latent_pass(x)

    def _target(self, y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        target = y.squeeze(-1)
        if mean.shape != target.shape:
            raise ValueError(
                "prediction and target shapes must match "
                "(output_dim should equal horizon): "
                f"{mean.shape} != {target.shape}"
            )
        return target

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    
    def get_layered_kl(self, x,y):
        return torch.Tensor([0]) # this will be implemented in a more advanced stage




    def compute_losses(self, x, y, prior=True):
        mean, logvar = self(x)
        loss = self.nllLoss(mean, self._target(y, mean), logvar.exp())
        return loss, 0 # need kl as well

    
    def _latent_pass(self, x, y=None, prior=True):
        if x.ndim == 2:
            x = x.unsqueeze(-1)


        x, _ = self.prior_state(x)
        
        mean, logvar = self.module(x)[:, -1, :].chunk(2, dim=-1)

        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

        x = self.linear(z)

        mean = self._to_output_shape(
            x[:, : self.config.output_dim * self.config.horizon]
        )
        logvar = self._to_output_shape(
            x[:, self.config.output_dim * self.config.horizon :]
        )
        return mean, logvar



    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        mean, logvar = self(x)
        loss = self.nllLoss(mean, self._target(y, mean), logvar.exp())
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device)
        y = y.to(self.device)
        mean, logvar = self(x)
        return float(self.loss(mean, self._target(y, mean), logvar.exp()))



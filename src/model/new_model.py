import torch
from torch import nn

from experiments.util import SeriesConfig

TDLGMConfig = SeriesConfig


def _resolve_num_heads(hidden_dim: int) -> int:
    for candidate in (8, 4, 2, 1):
        if hidden_dim % candidate == 0 and hidden_dim // candidate >= 16:
            return candidate
    return 1


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
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=layers,
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
            output_dim=2 * config.output_dim *config.horizon,
        )

        self.rnn = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=4,
            batch_first=True,
        )

        self.nllLoss = nn.GaussianNLLLoss()
        self.config = config




    

    def _to_output_shape(self, x: torch.Tensor) -> torch.Tensor:

        print("Output shape:", x.shape)
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("Input shape:", x.shape)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take the last time step's output
        x = self.model(x)


        mean, logvar = x.chunk(2, dim=-1)
        return self._to_output_shape(mean), self._to_output_shape(logvar)

    def _compute_losses(self, y, pred_mean, pred_logvar):
        # Reconstruction loss
        recon_loss = self.nllLoss(pred_mean, y.squeeze(-1), pred_logvar.exp())
        return (
            recon_loss,
            0.0,
        )  # We currently have no KL divergence for this simple model, so we return 0.0 for the KL loss.

    def compute_losses(self, x: torch.Tensor, y: torch.Tensor):
        mean, logvar = self.forward(x)
        rec, kl = self._compute_losses(y, mean, logvar)
        return float(rec), float(kl)

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        mean, logvar = self.forward(x)
        loss = self.nllLoss(mean, y.squeeze(-1), logvar.exp())
        loss.backward()
        optimizer.step()
        return float(loss)

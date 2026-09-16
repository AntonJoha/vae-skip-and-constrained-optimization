from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _imports():
    series_config = import_module("experiments.util").SeriesConfig
    vrnn_model = import_module("model").VRNN_Model
    return series_config, vrnn_model


def test_vrnn_forward_shapes_single_output():
    series_config, vrnn_model = _imports()
    config = series_config(
        input_dim=1,
        output_dim=1,
        hidden_dim=8,
        layers=1,
        seq_len=6,
        horizon=3,
    )
    model = vrnn_model(config)
    x = torch.randn(4, config.seq_len, config.input_dim)
    mean, logvar = model(x)
    assert mean.shape == (4, config.horizon)
    assert logvar.shape == (4, config.horizon)


def test_vrnn_compute_losses_prior_modes():
    series_config, vrnn_model = _imports()
    config = series_config(
        input_dim=1,
        output_dim=1,
        hidden_dim=8,
        layers=1,
        seq_len=5,
        horizon=2,
    )
    model = vrnn_model(config)
    x = torch.randn(3, config.seq_len, config.input_dim)
    y = torch.randn(3, config.horizon, config.output_dim)

    rec_post, kl_post = model.compute_losses(x, y, prior=False)
    rec_prior, kl_prior = model.compute_losses(x, y, prior=True)

    assert rec_post >= 0.0
    assert rec_prior >= 0.0
    assert kl_post >= 0.0
    assert kl_prior == pytest.approx(0.0, abs=1e-5)


def test_vrnn_train_step_and_mismatched_io_dims():
    series_config, vrnn_model = _imports()
    config = series_config(
        input_dim=1,
        output_dim=2,
        hidden_dim=8,
        layers=1,
        seq_len=4,
        horizon=2,
        beta=0.5,
    )
    model = vrnn_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, config.seq_len, config.input_dim)
    y = torch.randn(2, config.horizon, config.output_dim)

    loss = model.train_step(x, y, optimizer)
    assert torch.isfinite(torch.tensor(loss))

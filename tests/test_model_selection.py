from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _base_runtime():
    series_config = import_module("experiments.util").SeriesConfig
    return series_config(
        input_dim=1,
        output_dim=1,
        hidden_dim=8,
        layers=1,
        seq_len=4,
        horizon=2,
        vrnn=False,
        basic=False,
        upper=False,
        lower=False,
    )


@pytest.mark.parametrize(
    ("flags", "expected_name"),
    [
        ({"vrnn": True}, "vrnn"),
        ({"basic": True}, "basic"),
        ({"upper": True}, "tdlgm_upper"),
        ({"lower": True}, "tdlgm_lower"),
        ({}, "tdlgm_reg"),
    ],
)
def test_build_runtime_model_sets_model_name(flags, expected_name):
    main_module = import_module("experiments.main")
    runtime = _base_runtime()
    for key, value in flags.items():
        setattr(runtime, key, value)
    _model, _optimizer = main_module.build_runtime_model(runtime)
    assert runtime.model_name == expected_name

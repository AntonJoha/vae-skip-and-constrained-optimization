from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse(argv: list[str]):
    from experiments.cli import parse_args

    old = sys.argv
    try:
        sys.argv = ["experiments.main", *argv]
        return parse_args()
    finally:
        sys.argv = old


def test_parse_args_defaults():
    args = _parse([])
    assert args.vrnn is False
    assert args.basic is False
    assert args.upper is False
    assert args.lower is False


def test_parse_args_vrnn_enabled():
    args = _parse(["--vrnn"])
    assert args.vrnn is True
    assert args.basic is False
    assert args.upper is False
    assert args.lower is False


def test_parse_args_basic_enabled():
    args = _parse(["--basic"])
    assert args.basic is True
    assert args.vrnn is False
    assert args.upper is False
    assert args.lower is False


def test_parse_args_rejects_multiple_model_flags():
    with pytest.raises(SystemExit):
        _parse(["--vrnn", "--basic"])

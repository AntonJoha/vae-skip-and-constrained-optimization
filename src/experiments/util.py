import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )
    import optuna.logging

    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )


DATASET_PATH = Path(__file__).with_name("data").joinpath("shampoo_sales.csv")


@dataclass(slots=True)
class DataConfig:
    seq_len: int = 50
    batch_size: int = 8
    horizon: int = 1
    shampoo_code: bool = False
    reduced_dataset: float | None = None
    train_fraction: float = 0.8
    artifact_dir: str = "artifacts_dev/tdlgm"
    checkpoint_interval: int = 10
    early_stopping_patience: int = 10
    run_id: str | None = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    model_name: str = "tdlgm"


@dataclass(slots=True)
class BaselineConfig(DataConfig):
    input_dim: int = 1
    hidden_dim: int = 20
    latent_dim: int = 5
    output_dim: int = 1
    layers: int = 2

    learning_rate: float = 1e-3

    epochs: int = 80
    seed: int = 42
    device: str | None = None


@dataclass(slots=True)
class SeriesConfig(BaselineConfig):
    # Architecture overrides
    hidden_dim: int = 32
    latent_dim: int = 8
    tdlgm_layers: int = 2

    # Training overrides
    batch_size: int = 64
    learning_rate: float = 1e-3

    beta: float = 1e-3
    alpha: float = 1e-2
    weight_decay: float = 1e-5

    std: float = 0.2

    # Training/tuning
    tuning_trials: int = 100
    tuning_epochs: int = 20
    tune: bool = False

    # Misc
    verbose: bool = False
    baseline: bool = False
    use_old: bool = False
    attention: bool = False  ### KEPT HERE FOR BACKWARD COMPATIBILITY, SHOULD BE REMOVED


def checkpoint_payload(model: nn.Module, runtime: SeriesConfig) -> dict[str, object]:
    return {
        "config": asdict(runtime),
        "model_config": asdict(model.config),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_state_dict": model.state_dict(),
    }


def save_checkpoint(
    model: nn.Module, runtime: SeriesConfig, checkpoint_path: Path
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}_{runtime.run_id}.pt"
    )
    torch.save(checkpoint_payload(model, runtime), saved_path)
    return saved_path


def save_config(
    runtime: SeriesConfig, model: nn.Module, output_dir: Path, run_id: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"config_{run_id}.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "config": asdict(runtime),
                "model_config": asdict(model.config),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return config_path


def checkpoint_filename(suffix: str) -> str:
    return f"checkpoint_epoch{suffix}"


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[SeriesConfig, SeriesConfig, dict[str, torch.Tensor], str | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime = SeriesConfig(**checkpoint["config"])
    model_config = SeriesConfig(**checkpoint["model_config"])
    model_state = checkpoint["model_state_dict"]
    model_class = checkpoint.get("model_class")
    return runtime, model_config, model_state, model_class

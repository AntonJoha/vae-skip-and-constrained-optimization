from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data import get_scale_constant, make_dataloaders
from experiments.baseline import Baseline
from experiments.main import unpack_batch
from experiments.util import SeriesConfig, configure_logging, load_checkpoint
from model import Reg_Model, Upper_Model, VAE_Baseline_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

nll_loss = nn.GaussianNLLLoss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")


def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1) if x.ndim == 2 else x


def _mean_metric(values: list[torch.Tensor | float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved tDLGM checkpoint on the validation split."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("artifacts/tdlgm/checkpoint_epochfinal_20260807-105536.pt"),
        help="Path to a saved checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/eval"),
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def nll_position(
    mean: torch.Tensor, y: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    mean = _ensure_sequence(mean)
    logvar = _ensure_sequence(logvar)
    y = _ensure_sequence(y)
    return nll_loss(mean, y, logvar.exp()).mean(dim=(0, 2))


def fde_position(mean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean = _ensure_sequence(mean)
    y = _ensure_sequence(y)
    return torch.linalg.vector_norm(mean[:, -1, :] - y[:, -1, :], dim=-1).mean()


def ade_position(mean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean = _ensure_sequence(mean)
    y = _ensure_sequence(y)
    return torch.linalg.vector_norm(mean - y, dim=-1).mean()


def mse_position(mean: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean = _ensure_sequence(mean)
    y = _ensure_sequence(y)
    return mse_loss(mean, y).mean(dim=(0, 2))


def _evaluate_sequence_model(
    loader: DataLoader,
    forward_fn,
    loss_fn,
    scaler,
) -> dict:
    results = {
        "x": [],
        "x_scaled": [],
        "mean": [],
        "logvar": [],
        "mean_scaled": [],
        "logvar_scaled": [],
        "y": [],
        "y_scaled": [],
        "losses": [],
        "losses_position": [],
        "mse_losses": [],
        "mse_losses_position": [],
        "ade_losses_position": [],
        "ade_losses": [],
        "fde_losses_position": [],
        "fde_losses": [],
    }

    for batch in loader:
        x, y = unpack_batch(batch)
        mean, logvar = forward_fn(x, y)
        mean_scaled = scaler[0](mean)
        y_scaled = scaler[0](y)
        logvar_scaled = scaler[1](logvar)
        x_scaled = scaler[0](x)

        results["x"].append(x)
        results["x_scaled"].append(x_scaled)
        results["mean"].append(mean)
        results["logvar"].append(logvar)
        results["mean_scaled"].append(mean_scaled)
        results["logvar_scaled"].append(logvar_scaled)
        results["y"].append(y)
        results["y_scaled"].append(y_scaled)

        results["losses"].append(float(loss_fn(mean, y.squeeze(-1), logvar.exp())))
        results["losses_position"].append(nll_position(mean, y.squeeze(-1), logvar))
        results["mse_losses"].append(float(mse_loss(mean, y.squeeze(-1)).mean()))
        results["mse_losses_position"].append(mse_position(mean, y.squeeze(-1)))
        results["ade_losses_position"].append(ade_position(mean_scaled, y_scaled.squeeze(-1)))
        results["ade_losses"].append(float(ade_position(mean_scaled, y_scaled.squeeze(-1)).mean()))
        results["fde_losses_position"].append(fde_position(mean_scaled, y_scaled.squeeze(-1)))
        results["fde_losses"].append(float(fde_position(mean_scaled, y_scaled.squeeze(-1)).mean()))

    results["loss"] = _mean_metric(results["losses"])
    results["loss_position"] = _mean_metric(results["losses_position"])
    results["mse_loss"] = _mean_metric(results["mse_losses"])
    results["mse_loss_position"] = _mean_metric(results["mse_losses_position"])
    results["ade_loss_position"] = _mean_metric(results["ade_losses_position"])
    results["ade_loss"] = _mean_metric(results["ade_losses"])
    results["fde_loss_position"] = _mean_metric(results["fde_losses_position"])
    results["fde_loss"] = _mean_metric(results["fde_losses"])
    return results


@torch.no_grad()
def evaluate_baseline(model: nn.Module, loader: DataLoader, scaler) -> float:
    model.eval()
    return _evaluate_sequence_model(
        loader,
        forward_fn=lambda x, _y: model(x),
        loss_fn=model.loss,
        scaler=scaler,
    )


@torch.no_grad()
def evaluate_tdlgm(model: nn.Module, loader: DataLoader, scaler) -> float:
    logger.info("Evaluating tDLGM model...")

    model.eval()
    return _evaluate_sequence_model(
        loader,
        forward_fn=lambda x, y: model(x),
        loss_fn=model.nllLoss,
        scaler=scaler,
    )


def remove_pytorch(results: dict) -> dict:
    def remove_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: remove_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [remove_tensors(v) for v in obj]
        else:
            return obj

    return remove_tensors(results)


def _set_input_output_dim(runtime: SeriesConfig, loader: DataLoader) -> None:
    for x, y in loader:
        runtime.input_dim = x.shape[-1]
        runtime.output_dim = y.shape[-1]
        break


def _load_tdlgm_state_dict(model: nn.Module, model_state: dict) -> None:
    current_state = model.state_dict()
    compatible_state = {}
    legacy_mlp_key = re.compile(
        r"^(model|(?:downwards_list|upwards_list)\.\d+)\.(2|4)\.(weight|bias)$"
    )
    current_indices = {"2": "3", "4": "6"}

    for key, value in model_state.items():
        if key in current_state:
            compatible_state[key] = value
            continue

        match = legacy_mlp_key.fullmatch(key)
        if match:
            prefix, index, parameter = match.groups()
            current_key = f"{prefix}.{current_indices[index]}.{parameter}"
            if (
                current_key in current_state
                and current_state[current_key].shape == value.shape
            ):
                key = current_key
        compatible_state[key] = value

    model.load_state_dict(compatible_state)


def benchmark_model(args, model_path: Path) -> None:
    _runtime, model_config, model_state, _model_class = load_checkpoint(model_path)

    print(model_config)
    eval_config = replace(
        model_config,
        reduced_dataset=1,
        output_dim=model_config.horizon,
        batch_size=args.batch_size,
    )

    _, _, test_loader = make_dataloaders(eval_config)
    scaler = get_scale_constant(eval_config)
    _set_input_output_dim(eval_config, test_loader)

    res = None
    if model_config.model_name == "tdlgm":
        if model_config.upper:
            model = Upper_Model(model_config).to(device)
        elif model_config.vae_baseline:
            model = VAE_Baseline_Model(model_config).to(device)
        else:
            model = Reg_Model(model_config).to(device)

        _load_tdlgm_state_dict(model, model_state)
        _, _, test_loader = make_dataloaders(eval_config)
        res = evaluate_tdlgm(model, test_loader, scaler)
    elif model_config.model_name == "baseline":
        model = Baseline(model_config).to(device)
        model.load_state_dict(model_state)
        _, _, test_loader = make_dataloaders(eval_config)
        res = evaluate_baseline(model, test_loader, scaler)

    print(
        f"Model: {model_config.model_name}, Checkpoint: {model_path}, Test Loss: {res['loss']:.5f}, NLL Position Loss: {res['loss_position']}, Test MSE Loss: {res['mse_loss']:.5f} MSE Position Loss: {res['mse_loss_position']}"
    )
    print(
        f"Test ADE Position Loss: {res['ade_loss_position']:.5f}, Test FDE Position Loss: {res['fde_loss_position']:.5f}, Test ADE Loss: {res['ade_loss']:.5f}, Test FDE Loss: {res['fde_loss']:.5f}"
    )

    return remove_pytorch(res)


def save_results(output_dir: Path, checkpoint_path: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_results_{checkpoint_path.stem}.pt"
    results["checkpoint_path"] = str(checkpoint_path)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved evaluation results to %s", output_file)


def main() -> None:
    args = parse_args()
    args.batch_size = 1
    configure_logging(args.verbose)
    res = benchmark_model(args, args.checkpoint_path)
    save_results(args.output_dir, args.checkpoint_path, res)


if __name__ == "__main__":
    main()

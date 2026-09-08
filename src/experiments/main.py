from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from optuna.exceptions import TrialPruned
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.data import make_dataloaders
from experiments.util import (
    SeriesConfig,
    checkpoint_filename,
    configure_logging,
    save_checkpoint,
    save_config,
)
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)


def unpack_batch(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if y.ndim == 2:
        y = y.unsqueeze(-1)

    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x, y = unpack_batch(batch)
        mean, logvar, *_ = model(x)
        log.info("Mean shape: %s, Logvar shape: %s, Y shape: %s", mean.shape, logvar.shape, y.shape)

        losses.append(float(model.nllLoss(mean, y.squeeze(-1), logvar.exp())))
    model.train()
    return sum(losses) / max(1, len(losses))


def build_runtime_model(runtime: SeriesConfig) -> tuple[nn.Module, Adam]:
    model = Model(runtime).to(device)
    optimizer = Adam(model.parameters(), lr=runtime.learning_rate)
    if runtime.verbose:
        log.info(
            "Parameters: %s",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    return model, optimizer


def _set_input_output_dim(runtime: SeriesConfig, loader: DataLoader) -> None:
    for batch in loader:
        x, y = unpack_batch(batch)
        runtime.input_dim = x.shape[-1]
        runtime.output_dim = y.shape[-1]
        break


def train_model(
    runtime: SeriesConfig,
    epochs: int | None = None,
    trial: optuna.Trial | None = None,
    save_to: Path | None = None,
) -> tuple[float, float]:
    torch.manual_seed(runtime.seed)
    runtime = replace(runtime, output_dim=runtime.horizon)

    train_loader, val_loader, _test_loader = make_dataloaders(runtime)

    _set_input_output_dim(runtime, train_loader)

    model, optimizer = build_runtime_model(runtime)
    train_epochs = runtime.epochs if epochs is None else epochs
    checkpoint_interval = max(1, runtime.checkpoint_interval)
    early_stopping_patience = runtime.early_stopping_patience
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    before = evaluate(model, val_loader)
    best_val = before
    epochs_without_improvement = 0
    if runtime.verbose:
        log.info("Validation loss before training: %.5f", before)

    if save_to is not None:
        config_path = save_config(runtime, model, save_to, timestamp)
        if runtime.verbose:
            log.info("Saved configuration to %s", config_path)

    model.train()
    for epoch in range(train_epochs):
        log.info("Starting epoch %03d/%03d", epoch + 1, train_epochs)
        epoch_losses = []
        recon_loss = kl_loss = consistency = 0.0
        recon_loss_p = kl_loss_p = consistency_p = 0.0
        model.epoch = epoch

        for batch in train_loader:
            x, y = unpack_batch(batch)
            epoch_losses.append(model.train_step(x, y, optimizer))
            t_recon_loss, t_kl_loss = model.compute_losses(
                x,
                y,
            )
            t_recon_loss_p, t_kl_loss_p = model.compute_losses(
                x,
                y,
            )
            recon_loss += t_recon_loss
            kl_loss += t_kl_loss
            recon_loss_p += t_recon_loss_p
            kl_loss_p += t_kl_loss_p

        val_loss = evaluate(model, val_loader)

        if runtime.verbose:
            train_batches = max(1, len(train_loader))
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            recon_loss /= train_batches
            kl_loss /= train_batches
            consistency /= train_batches
            recon_loss_p /= train_batches
            kl_loss_p /= train_batches
            consistency_p /= train_batches

            log.info("========== Epoch %03d =========", epoch + 1)
            log.info(" Train loss: %.5f: NLL on val set: %.5f", mean_loss, val_loss)
            log.info(" Posterior: NLL %.5f: kl_loss %.5f:", recon_loss, kl_loss)
            log.info(" Prior: NLL %.5f: kl_loss %.5f:", recon_loss_p, kl_loss_p)

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            if save_to is not None:
                saved_path = save_checkpoint(
                    model,
                    runtime,
                    save_to / checkpoint_filename("best"),
                )
                if runtime.verbose:
                    log.info("Saved checkpoint to %s", saved_path)
        else:
            epochs_without_improvement += 1

        if trial is not None:
            log.info(
                "Trial %d: Epoch %03d: Validation loss %.5f",
                trial.number,
                epoch + 1,
                val_loss,
            )
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()

        if save_to is not None and (
            (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == train_epochs
        ):
            checkpoint_path = save_to / checkpoint_filename(f"{epoch + 1:04d}")
            saved_path = save_checkpoint(model, runtime, checkpoint_path)
            if runtime.verbose:
                log.info("Saved checkpoint to %s", saved_path)

        if epochs_without_improvement >= early_stopping_patience and trial is None:
            if runtime.verbose:
                log.info(
                    "Early stopping after %d epochs without val NLL improvement.",
                    early_stopping_patience,
                )
            break

    after = evaluate(model, val_loader)
    if runtime.verbose:
        log.info("Validation loss after training: %.5f and before %.5f", after, before)
    if trial is None and after >= before:
        log.warning(
            "Validation loss did not improve: before=%.5f after=%.5f",
            before,
            after,
        )

    if save_to is not None:
        saved_path = save_checkpoint(
            model,
            runtime,
            save_to / checkpoint_filename("final"),
        )
        if runtime.verbose:
            log.info("Saved checkpoint to %s", saved_path)
    return before, after, best_val


def tune_hyperparameters(base_runtime: SeriesConfig) -> SeriesConfig:
    def objective(trial: optuna.Trial) -> float:
        runtime = replace(
            base_runtime,
            hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512]),
            latent_dim=trial.suggest_categorical(
                "latent_dim",
                [8, 16, 32, 64, 128],
            ),
            tdlgm_layers=trial.suggest_int(
                "tdlgm_layers",
                1,
                4,
            ),
            layers=trial.suggest_int("layers", 1, 3),
            beta=trial.suggest_float("beta", 1e-3, 1, log=True),
            alpha=trial.suggest_float("alpha", 1 + 1e-9, 1 + 1.1e-3, log=True),
            learning_rate=trial.suggest_float(
                "learning_rate",
                1e-5,
                3e-3,
                log=True,
            ),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            weight_decay=trial.suggest_float(
                "weight_decay",
                1e-8,
                1e-2,
                log=True,
            ),
        )
        _, _, best = train_model(runtime, epochs=runtime.tuning_epochs, trial=trial)
        return best

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base_runtime.seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    log.info("Best hyperparameters: %s", study.best_trial.params)
    log.info("Best validation loss during tuning: %.5f", study.best_value)
    return replace(base_runtime, **study.best_trial.params)


def train(base_runtime: SeriesConfig) -> Path:
    torch.manual_seed(base_runtime.seed)
    if base_runtime.verbose:
        log.info(
            "Starting training with %s.", "tuning" if base_runtime.tune else "no tuning"
        )

    runtime = tune_hyperparameters(base_runtime) if base_runtime.tune else base_runtime
    if base_runtime.verbose and not base_runtime.tune:
        log.info("Skipping hyperparameter tuning.")

    artifact_dir = Path(base_runtime.artifact_dir)
    train_model(runtime, save_to=artifact_dir)
    return artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train our model on time series data.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline training instead of our model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--horizon", type=int, default=10, help="Forecast horizon."
        )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_runtime = SeriesConfig(**vars(args))

    configure_logging(args.verbose)

    if args.baseline:
        from experiments.baseline import baseline_train

        baseline_train(base_runtime)
        return

    train(base_runtime)


if __name__ == "__main__":
    main()

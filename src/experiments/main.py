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
    should_stop_training,
)
from model import VRNN, Basic, Lower_Model, Reg_Model, Upper_Model, VAE_Baseline_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)

MODEL_FACTORIES = (
    (lambda runtime: runtime.vae_baseline, VAE_Baseline_Model),
    (lambda runtime: runtime.vrnn, VRNN),
    (lambda runtime: runtime.basic, Basic),
    (lambda runtime: runtime.upper, Upper_Model),
    (lambda runtime: runtime.lower, Lower_Model),
)


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
        t_recon_loss, _ = model.compute_losses(x, y)
        losses.append(t_recon_loss)
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate_posterior(model, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x, y = unpack_batch(batch)
        t_recon_loss_q, _ = model.compute_losses(
            x,
            y,
            prior=False,
        )
        losses.append(t_recon_loss_q)
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate_kl(model, loader: DataLoader) -> float:
    model.eval()
    kl_losses = None
    count = 0
    for batch in loader:
        x, y = unpack_batch(batch)
        temp = model.get_layered_kl(x, y)
        if kl_losses is None:
            kl_losses = temp
        else:
            kl_losses += temp
        count += 1
    model.train()
    return kl_losses / max(1, count)


def build_runtime_model(runtime: SeriesConfig) -> tuple[nn.Module, Adam]:
    for predicate, factory in MODEL_FACTORIES:
        if predicate(runtime):
            model = factory(runtime).to(device)
            break
    else:
        model = Reg_Model(runtime).to(device)

    log.info("Initializing model with Adam and lr = %.5f", runtime.learning_rate)
    
    inner_param = model.inner_parameters()
    inner_optimizer = None
    if inner_param is not None:
        inner_optimizer = Adam(model.inner_parameters(), lr=runtime.learning_rate)

    middle_param = model.middle_parameters()
    middle_optimizer = None
    if middle_param is not None:
        middle_optimizer = Adam(model.middle_parameters(), lr=runtime.learning_rate)


    if runtime.verbose:
        log.info(
            "Parameters: %s",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    model.compile()
    return model, inner_optimizer, middle_optimizer


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

    model, inner_optimizer, middle_optimizer = build_runtime_model(runtime)
    
    scheduler_inner = None
    if inner_optimizer is not None:
        scheduler_inner = torch.optim.lr_scheduler.ReduceLROnPlateau(
            inner_optimizer,
            mode='min',
            factor=0.2,
            patience=10,
        )
    
    scheduler_middle = None
    if middle_optimizer is not None:
        scheduler_middle = torch.optim.lr_scheduler.ReduceLROnPlateau(
            middle_optimizer,
            mode='min',
            factor=0.1,
            patience=10,
        )

    train_epochs = runtime.epochs if epochs is None else epochs
    checkpoint_interval = max(1, runtime.checkpoint_interval)
    early_stopping_patience = 20
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    before = evaluate(model, val_loader)
    best_val = before
    epochs_without_improvement = 0
    stopped_due_to_divergence = False
    if runtime.verbose:
        log.info("Validation loss before training: %.5f", before)
    if reason := should_stop_training(before, before):
        log.warning("Stopping before training because %s: %.5f", reason, before)
        if trial is not None:
            raise TrialPruned()
        return before, before, best_val

    if save_to is not None:
        config_path = save_config(runtime, model, save_to, timestamp)
        if runtime.verbose:
            log.info("Saved configuration to %s", config_path)

    model.train()
    for epoch in range(train_epochs):
        log.info("Starting epoch %03d/%03d", epoch + 1, train_epochs)
        epoch_losses = []
        model.set_epoch(epoch)

        for batch in train_loader:
            x, y = unpack_batch(batch)
            epoch_losses.append(model.train_step(x, y, middle_optimizer, inner_optimizer))

        model.outer_train_step(train_loader)

        val_loss = evaluate(model, val_loader)
        if runtime.verbose:
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            val_posterior = evaluate_posterior(model, val_loader)
            post = evaluate_posterior(model, train_loader)
            prior = evaluate(model, train_loader)
            layered_kl = evaluate_kl(model, train_loader)
            log.info("========== Epoch %03d =========", epoch + 1)
            log.info(
                " Train loss: %.5f: NLL on val set: %.5f, Posterior NLL on val: %.5f",
                mean_loss,
                val_loss,
                val_posterior,
            )
            log.info(
                " Posterior NLL on train: %.5f",
                post,
            )
            log.info(" Prior NLL on train: %.5f", prior)
            log.info(" Layered KL: %s", layered_kl)
        
        if scheduler_middle is not None:
            scheduler_middle.step(val_loss)
        if scheduler_inner is not None:
            scheduler_inner.step(val_loss)

        if reason := should_stop_training(before, val_loss):
            log.warning(
                "Stopping early at epoch %03d because %s: %.5f",
                epoch + 1,
                reason,
                val_loss,
            )
            if trial is not None:
                raise TrialPruned()
            stopped_due_to_divergence = True
            break

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

    if save_to is not None and not stopped_due_to_divergence:
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
            hidden_dim=trial.suggest_categorical(
                "hidden_dim",
                [ 32, 64, 128, 256, 512],
            ),
            layers=trial.suggest_int(
                "layers",
                2,
                6,
            ),
            batch_size=trial.suggest_categorical(
                "batch_size",
                [16, 32, 64, 128],
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                1e-6,
                5e-2,
                log=True,
            ),
            lr_lambda=trial.suggest_float(
                "lr_lambda",
                1e-5,
                1e-1,
                log=True,
            ),
            rho=trial.suggest_float(
                "rho",
                1e-2,
                1e1,
                log=True,
            ),
            reduction_threshold=trial.suggest_float(
                "reduction_threshold",
                0.8,
                0.99,
            ),
            weight_decay=trial.suggest_float(
                "weight_decay",
                1e-7,
                1e-2,
                log=True,
            ),
            beta=trial.suggest_float(
                "beta",
                1e-2,
                1),
            lambda_min=trial.suggest_categorical(
                "lambda_min",
                [.5, 1, 5]
            ),
            skip_connection=trial.suggest_categorical("skip_connection", [True, False])
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
        "--epochs", type=int, default=1000, help="Number of training epochs."
    )
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon.")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning.")
    parser.add_argument("--skip_connection", action="store_true", help="Enable hyperparameter tuning.")
    parser.add_argument("--upper", action="store_true", help="Enable the upper bound KL Model")
    parser.add_argument("--lower", action="store_true", help="Enable the lower bound KL Model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Fix the learning rate")
    parser.add_argument("--reduced_dataset", type=float, default=1.0, help="Reduce the used dataset")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--reverse", action="store_true",  default=False)
    parser.add_argument("--vae-baseline", action="store_true", default=False)

    parser.add_argument("--lr_lambda", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_runtime = SeriesConfig(**vars(args))

    configure_logging(args.verbose)
    log.info("Using %s for training", str(device))

    if args.baseline:
        from experiments.baseline import baseline_train

        baseline_train(base_runtime)
        return

    train(base_runtime)




if __name__ == "__main__":
    main()

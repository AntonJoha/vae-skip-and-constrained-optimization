from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn
from torch.utils.data import DataLoader

from data.data import make_dataloaders
from experiments.util import (
    BaselineConfig,
    SeriesConfig,
    checkpoint_filename,
    save_checkpoint,
    save_config,
    should_stop_training,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Baseline(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.linear = nn.Linear(
            config.hidden_dim, config.output_dim * 2 * config.horizon
        )
        self.loss = nn.GaussianNLLLoss()
        self.config = config


    def _to_output_shape(self, x):
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = self.linear(x)[:, -1, :]

        mean = self._to_output_shape(
            x[:, : self.config.output_dim * self.config.horizon]
        )
        logvar = self._to_output_shape(
            x[:, self.config.output_dim * self.config.horizon :]
        )
        return mean, logvar

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

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        mean, logvar = self(x)
        loss = self.loss(mean, self._target(y, mean), logvar.exp())
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device)
        y = y.to(self.device)
        mean, logvar = self(x)
        return float(self.loss(mean, self._target(y, mean), logvar.exp()))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = [model.get_loss(x, y) for x, y in loader]
    return sum(losses) / max(1, len(losses))


def _set_input_output_dim(runtime: SeriesConfig, loader: DataLoader) -> None:
    for x, y in loader:
        runtime.input_dim = x.shape[-1]
        runtime.output_dim = y.shape[-1]
        break


def train_model(
    runtime: BaselineConfig,
    epochs: int | None = None,
    trial: optuna.Trial | None = None,
    save_to: Path | None = None,
) -> tuple[float, float]:
    torch.manual_seed(runtime.seed)
    runtime = replace(runtime, output_dim=runtime.horizon)

    train_loader, val_loader, _test_loader = make_dataloaders(runtime)
    _set_input_output_dim(runtime, train_loader)

    model = Baseline(runtime).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=runtime.learning_rate)
    train_epochs = runtime.epochs if epochs is None else epochs
    checkpoint_interval = max(1, runtime.checkpoint_interval)
    early_stopping_patience = runtime.early_stopping_patience
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    before = evaluate(model, val_loader)
    best_val = before
    epochs_without_improvement = 0
    stopped_due_to_divergence = False
    if runtime.verbose:
        logger.info("Validation loss before training: %.5f", before)
    if reason := should_stop_training(before, before):
        logger.warning("Stopping before training because %s: %.5f", reason, before)
        if trial is not None:
            raise TrialPruned()
        return before, before

    if save_to is not None:
        config_path = save_config(runtime, model, save_to, timestamp)
        if runtime.verbose:
            logger.info("Saved configuration to %s", config_path)

    model.train()
    for epoch in range(train_epochs):
        epoch_losses = []
        for x, y in train_loader:
            epoch_losses.append(model.train_step(x, y, optimizer))

        val_loss = evaluate(model, val_loader)
        if reason := should_stop_training(before, val_loss):
            logger.warning(
                "Stopping early at epoch %03d because %s: %.5f",
                epoch + 1,
                reason,
                val_loss,
            )
            if trial is not None:
                raise TrialPruned()
            stopped_due_to_divergence = True
            break

        if runtime.verbose:
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            logger.info(
                "Epoch %03d: Loss %.5f  Val loss %.5f",
                epoch + 1,
                mean_loss,
                val_loss,
            )

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
                    logger.info("Saved checkpoint to %s", saved_path)
        else:
            epochs_without_improvement += 1

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()

        if save_to is not None and (
            (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == train_epochs
        ):
            saved_path = save_checkpoint(
                model,
                runtime,
                save_to / checkpoint_filename(f"{epoch + 1:04d}"),
            )
            if runtime.verbose:
                logger.info("Saved checkpoint to %s", saved_path)

        if epochs_without_improvement >= early_stopping_patience:
            if runtime.verbose:
                logger.info(
                    "Early stopping after %d epochs without val NLL improvement.",
                    early_stopping_patience,
                )
            #break

    after = evaluate(model, val_loader)
    if runtime.verbose:
        logger.info("Validation loss after training: %.5f", after)
    if trial is None and after >= before:
        logger.warning(
            "Validation loss did not improve: before=%.5f after=%.5f",
            before,
            after,
        )

    if save_to is not None and not stopped_due_to_divergence:
        saved_path = save_checkpoint(
            model, runtime, save_to / checkpoint_filename("final")
        )
        if runtime.verbose:
            logger.info("Saved checkpoint to %s", saved_path)
    return before, after


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
                1,
                4,
            ),
            batch_size=trial.suggest_categorical(
                "batch_size",
                [32, 64, 128],
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                1e-5,
                5e-3,
                log=True,
            ),
            weight_decay=trial.suggest_float(
                "weight_decay",
                1e-7,
                1e-2,
                log=True,
            ),
        )
        _, after = train_model(runtime, epochs=runtime.tuning_epochs, trial=trial)
        return after

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=base_runtime.seed)
    )
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    if base_runtime.verbose:
        logger.info("Best hyperparameters: %s", study.best_trial.params)
        logger.info("Best validation loss during tuning: %.5f", study.best_value)
    return replace(base_runtime, **study.best_trial.params)


def baseline_train(runtime: SeriesConfig) -> Path:
    torch.manual_seed(runtime.seed)
    runtime.model_name = "baseline"

    runtime = tune_hyperparameters(runtime) if runtime.tune else runtime
    artifact_dir = Path(runtime.artifact_dir)
    train_model(runtime, save_to=artifact_dir)
    return artifact_dir
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
from optuna.exceptions import TrialPruned
from torch import nn
from torch.utils.data import DataLoader

from data.data import make_dataloaders
from experiments.util import (
    BaselineConfig,
    SeriesConfig,
    checkpoint_filename,
    save_checkpoint,
    save_config,
    should_stop_training,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Baseline(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.layers,
            batch_first=True,
        )
        self.linear = nn.Linear(
            config.hidden_dim, config.output_dim * 2 * config.horizon
        )
        self.loss = nn.GaussianNLLLoss()
        self.config = config


    def _to_output_shape(self, x):
        x = x.view(x.size(0), self.config.horizon, self.config.output_dim)
        return x.squeeze(-1) if self.config.output_dim == 1 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = self.linear(x)[:, -1, :]

        mean = self._to_output_shape(
            x[:, : self.config.output_dim * self.config.horizon]
        )
        logvar = self._to_output_shape(
            x[:, self.config.output_dim * self.config.horizon :]
        )
        return mean, logvar

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

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        optimizer.zero_grad()
        x = x.to(self.device)
        y = y.to(self.device)
        mean, logvar = self(x)
        loss = self.loss(mean, self._target(y, mean), logvar.exp())
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def get_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device)
        y = y.to(self.device)
        mean, logvar = self(x)
        return float(self.loss(mean, self._target(y, mean), logvar.exp()))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = [model.get_loss(x, y) for x, y in loader]
    return sum(losses) / max(1, len(losses))


def _set_input_output_dim(runtime: SeriesConfig, loader: DataLoader) -> None:
    for x, y in loader:
        runtime.input_dim = x.shape[-1]
        runtime.output_dim = y.shape[-1]
        break


def train_model(
    runtime: BaselineConfig,
    epochs: int | None = None,
    trial: optuna.Trial | None = None,
    save_to: Path | None = None,
) -> tuple[float, float]:
    torch.manual_seed(runtime.seed)
    runtime = replace(runtime, output_dim=runtime.horizon)

    train_loader, val_loader, _test_loader = make_dataloaders(runtime)
    _set_input_output_dim(runtime, train_loader)

    model = Baseline(runtime).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=runtime.learning_rate)
    train_epochs = runtime.epochs if epochs is None else epochs
    checkpoint_interval = max(1, runtime.checkpoint_interval)
    early_stopping_patience = runtime.early_stopping_patience
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    before = evaluate(model, val_loader)
    best_val = before
    epochs_without_improvement = 0
    stopped_due_to_divergence = False
    if runtime.verbose:
        logger.info("Validation loss before training: %.5f", before)
    if reason := should_stop_training(before, before):
        logger.warning("Stopping before training because %s: %.5f", reason, before)
        if trial is not None:
            raise TrialPruned()
        return before, before

    if save_to is not None:
        config_path = save_config(runtime, model, save_to, timestamp)
        if runtime.verbose:
            logger.info("Saved configuration to %s", config_path)

    model.train()
    for epoch in range(train_epochs):
        epoch_losses = []
        for x, y in train_loader:
            epoch_losses.append(model.train_step(x, y, optimizer))

        val_loss = evaluate(model, val_loader)
        if reason := should_stop_training(before, val_loss):
            logger.warning(
                "Stopping early at epoch %03d because %s: %.5f",
                epoch + 1,
                reason,
                val_loss,
            )
            if trial is not None:
                raise TrialPruned()
            stopped_due_to_divergence = True
            break

        if runtime.verbose:
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            logger.info(
                "Epoch %03d: Loss %.5f  Val loss %.5f",
                epoch + 1,
                mean_loss,
                val_loss,
            )

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
                    logger.info("Saved checkpoint to %s", saved_path)
        else:
            epochs_without_improvement += 1

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()

        if save_to is not None and (
            (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == train_epochs
        ):
            saved_path = save_checkpoint(
                model,
                runtime,
                save_to / checkpoint_filename(f"{epoch + 1:04d}"),
            )
            if runtime.verbose:
                logger.info("Saved checkpoint to %s", saved_path)

        if epochs_without_improvement >= early_stopping_patience:
            if runtime.verbose:
                logger.info(
                    "Early stopping after %d epochs without val NLL improvement.",
                    early_stopping_patience,
                )
            #break

    after = evaluate(model, val_loader)
    if runtime.verbose:
        logger.info("Validation loss after training: %.5f", after)
    if trial is None and after >= before:
        logger.warning(
            "Validation loss did not improve: before=%.5f after=%.5f",
            before,
            after,
        )

    if save_to is not None and not stopped_due_to_divergence:
        saved_path = save_checkpoint(
            model, runtime, save_to / checkpoint_filename("final")
        )
        if runtime.verbose:
            logger.info("Saved checkpoint to %s", saved_path)
    return before, after


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
                1,
                4,
            ),
            batch_size=trial.suggest_categorical(
                "batch_size",
                [32, 64, 128],
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                1e-5,
                5e-3,
                log=True,
            ),
            weight_decay=trial.suggest_float(
                "weight_decay",
                1e-7,
                1e-2,
                log=True,
            ),
        )
        _, after = train_model(runtime, epochs=runtime.tuning_epochs, trial=trial)
        return after

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=base_runtime.seed)
    )
    study.optimize(objective, n_trials=base_runtime.tuning_trials)

    if base_runtime.verbose:
        logger.info("Best hyperparameters: %s", study.best_trial.params)
        logger.info("Best validation loss during tuning: %.5f", study.best_value)
    return replace(base_runtime, **study.best_trial.params)


def baseline_train(runtime: SeriesConfig) -> Path:
    torch.manual_seed(runtime.seed)
    runtime.model_name = "baseline"

    runtime = tune_hyperparameters(runtime) if runtime.tune else runtime
    artifact_dir = Path(runtime.artifact_dir)
    train_model(runtime, save_to=artifact_dir)
    return artifact_dir

from __future__ import annotations

import argparse


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
    parser.add_argument(
        "--tune", action="store_true", help="Enable hyperparameter tuning."
    )
    parser.add_argument(
        "--skip_connection", action="store_true", help="Enable hyperparameter tuning."
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--upper", action="store_true", help="Enable the upper bound KL Model"
    )
    model_group.add_argument(
        "--lower", action="store_true", help="Enable the lower bound KL Model"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Fix the learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Fix the weight decay"
    )
    model_group.add_argument(
        "--basic",
        action="store_true",
        help="Use the basic model instead of the TDLGM model",
    )
    model_group.add_argument("--vrnn", action="store_true", help="Use the VRNN model")
    return parser.parse_args()

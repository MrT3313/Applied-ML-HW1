"""The training logic for one config, with no knowledge of W&B.

train() scores a config with cross validation and is what every sweep run calls. fit() fits the same config
once on all of its rows and is what entrypoints/promote.py calls for the run that was chosen.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.data import complete_rows
from src.evaluate import evaluate
from src.model import build_pipeline
from src.utils.make_folds import CVFolds


@dataclass
class TrainResult:
    """What train() hands back, in the two shapes W&B records.

    Attributes:
        metrics: One number per run (rmse_mean, rmse_sd, mae_mean, r2_mean). Becomes the run's summary.
        fold_metrics: One dict per fold (fold, fold/rmse, fold/mae, fold/r2). Becomes the run's history.
    """
    metrics: dict[str, float]
    fold_metrics: list[dict]


def summarize(scores: dict[str, np.ndarray]) -> TrainResult:
    """Turn per-fold score arrays into the run's summary numbers and one row per fold.

    The per-fold keys are named "fold/<metric>" so they cannot be mistaken for the per-run means, and so
    W&B groups their charts into one "fold" section.
    """
    metrics = {
        "rmse_mean": float(np.mean(scores["rmse"])),
        "rmse_sd": float(np.std(scores["rmse"], ddof=1)),
        "mae_mean": float(np.mean(scores["mae"])),
        "r2_mean": float(np.mean(scores["r2"])),
    }
    fold_metrics = [
        {"fold": fold, **{f"fold/{metric}": float(values[fold]) for metric, values in scores.items()}}
        for fold in range(len(scores["rmse"]))
    ]
    return TrainResult(metrics=metrics, fold_metrics=fold_metrics)


def usable_rows(config: dict, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """The rows the config trains on: only the complete rows with "drop", every row when imputing."""
    if config["data_missing"] == "drop":
        return complete_rows(X, y)
    return X, y


def train(config: dict, X: pd.DataFrame, y: pd.Series, folds: CVFolds) -> TrainResult:
    """Score the config's model with cross validation. Nothing is kept from the fitted models.

    The validation rows are the same complete rows for every config. Only the training side changes: with
    an "impute_..." value each fold also trains on its incomplete rows, filled by an imputer fit on that
    fold alone.

    Args:
        config: A validated run config.
        X: Every feature row of the training data, incomplete rows included.
        y: Target values, with the same index as X.
        folds: Folds from make_folds, built on that same full X.

    Returns:
        The per-run and per-fold scores.
    """
    X, y = usable_rows(config, X, y)

    pipeline = build_pipeline(config)
    scores = evaluate(pipeline, X, y, folds)
    return summarize(scores)


def fit(config: dict, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    """Fit the config's model once on every usable training row, the same rows train() uses.

    Cross validation only ever fits on part of the data. This is the model that gets saved and used.

    Args:
        config: A validated run config.
        X: Every feature row of the training data, incomplete rows included.
        y: Target values, with the same index as X.

    Returns:
        The fitted pipeline.
    """
    X, y = usable_rows(config, X, y)
    return build_pipeline(config).fit(X, y)

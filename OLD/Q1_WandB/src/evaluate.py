"""Scoring. evaluate() cross validates an unfitted pipeline, score() scores one fitted model once.

The metrics are the entries of SCORERS in src/utils/scorers.py. Add one there and it is computed here and
logged per fold by entrypoints/run.py with no other change.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

from src.utils.make_folds import CVFolds
from src.utils.scorers import GREATER_IS_BETTER, SCORERS


def evaluate(pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series, folds: CVFolds) -> dict[str, np.ndarray]:
    """Score a pipeline on every validation fold. Used by every training run.

    Args:
        pipeline: Unfitted model or pipeline. A fresh copy is fit on each fold's training rows.
        X: Feature rows the pipeline may train on. Must contain every validation row of the folds.
        y: Target values, with the same index as X.
        folds: Folds from make_folds.

    Returns:
        One array of per-fold scores per metric in SCORERS, in fold order, errors positive.
    """
    scores = cross_validate(pipeline, X, y, scoring=SCORERS, cv=folds.positions_for(X), n_jobs=-1)
    # a greater_is_better=False scorer returns the negated metric, so flip only those back
    return {
        metric: scores[f"test_{metric}"] if GREATER_IS_BETTER[metric] else -scores[f"test_{metric}"]
        for metric in SCORERS
    }


def score(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Score an already fitted model on one set of rows. Used for the final test set evaluation.

    Args:
        model: Fitted model or pipeline.
        X: Feature rows to predict.
        y: Their true target values.

    Returns:
        One number per metric in SCORERS, errors positive.
    """
    return {
        metric: float(scorer(model, X, y)) * (1 if GREATER_IS_BETTER[metric] else -1)
        for metric, scorer in SCORERS.items()
    }

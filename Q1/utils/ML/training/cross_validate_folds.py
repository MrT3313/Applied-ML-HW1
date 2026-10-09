import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline

from Q1.utils.ML.scoring.scorers import GREATER_IS_BETTER
from Q1.utils.ML.training.complete_row_folds import Folds


def cross_validate_folds(
	pipeline: Pipeline,
	*,
	X: pd.DataFrame,
	y: pd.Series,
	cv: BaseCrossValidator | Folds,
	scorers: dict,
	return_estimator: bool = False,
) -> dict[str, np.ndarray | list]:
	"""Score a pipeline on every validation fold with cross_validate.

	Args:
		pipeline: Model or pipeline to evaluate. It is cloned and refit on each training fold.
		X: Feature data.
		y: Target values.
		cv: A splitter, or a list of (train, validation) positions from complete_row_folds or repeated_folds.
		scorers: Scorers keyed by metric name. Every name must be a key of GREATER_IS_BETTER.
		return_estimator: Also return each fold's fitted estimator (ex to read a search's best_params_).

	Returns:
		One array of per-fold scores per metric, in fold order, each in its natural units (errors positive).
		With return_estimator, also "estimator": the fitted estimators in the same fold order.
	"""
	scores = cross_validate(
		pipeline, X, y,
		scoring=scorers,
		cv=cv,
		n_jobs=-1,
		return_estimator=return_estimator,
	)
	# a greater_is_better=False scorer returns the negated metric, so flip only those back
	fold_scores = {
		metric: scores[f"test_{metric}"] if GREATER_IS_BETTER[metric] else -scores[f"test_{metric}"]
		for metric in scorers
	}
	if return_estimator:
		fold_scores["estimator"] = list(scores["estimator"])
	return fold_scores

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

from OLD.utils.scoring.scorers import GREATER_IS_BETTER, SCORERS
from OLD.utils.pipelines.make_folds import CVFolds
from OLD.utils.pipelines.seed_pipeline import seed_pipeline


def evaluate_pipeline(
	pipeline: BaseEstimator,
	*,
	X: pd.DataFrame,
	y: pd.Series,
	folds: CVFolds,
	return_estimator: bool = False,
	seed: int = 42,
) -> dict[str, np.ndarray | list]:
	"""Seed a pipeline and score it on every validation fold with cross_validate and SCORERS.

	Args:
		pipeline: Model or pipeline to evaluate. It is cloned, seeded with seed_pipeline and refit on each
			training fold.
		X: Feature data. Its rows are matched to the folds by index label.
		y: Target values, with the same index as X.
		folds: Folds from make_folds.
		return_estimator: Also return each fold's fitted estimator (ex to read a search's best_params_).
		seed: Value seed_pipeline gives to every random_state in the pipeline.

	Returns:
		One array of per-fold scores per metric, in fold order, each in its natural units (errors positive).
		Also "validation_rows": the validation index labels of each fold, and "n_train" and "n_test": the
		training and validation row counts of each fold. With return_estimator, also "estimator": the fitted
		estimators in the same fold order.
	"""
	assert y.index.equals(X.index), "y and X must have the same index"

	seeded = seed_pipeline(pipeline, seed=seed)
	positions = folds.positions_for(X)
	scores = cross_validate(
		seeded, X, y,
		scoring=SCORERS,
		cv=positions,
		n_jobs=-1,
		return_estimator=return_estimator,
	)
	# a greater_is_better=False scorer returns the negated metric, so flip only those back
	fold_scores = {
		metric: scores[f"test_{metric}"] if GREATER_IS_BETTER[metric] else -scores[f"test_{metric}"]
		for metric in SCORERS
	}
	fold_scores["validation_rows"] = [X.index[test] for _, test in positions]
	fold_scores["n_train"] = np.array([len(train) for train, _ in positions])
	fold_scores["n_test"] = np.array([len(test) for _, test in positions])
	if return_estimator:
		fold_scores["estimator"] = list(scores["estimator"])
	return fold_scores

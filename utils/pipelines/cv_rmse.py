import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_validate

# Each fold: (positions of its training rows, positions of its validation rows)
Folds = list[tuple[np.ndarray, np.ndarray]]


def final_model(pipeline: BaseEstimator) -> BaseEstimator:
	"""The model at the end of a fitted pipeline from build_pipeline (or of a plain Pipeline, or the model itself)."""
	if isinstance(pipeline, TransformedTargetRegressor):
		pipeline = pipeline.regressor_
	return pipeline[-1] if hasattr(pipeline, 'steps') else pipeline


def cv_rmse(*, pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series, folds: Folds) -> pd.DataFrame:
	"""Cross validate a pipeline: fit it on the training rows of each fold and score its RMSE on the training and on
	the validation rows.

	Every pipeline scored on the same folds is scored on the same rows, so fold k of one result can be compared with
	fold k of another (see STATISTICS.fold_difference).

	Args:
		pipeline: The pipeline to cross validate. It is cloned for each fold.
		X: The attributes.
		y: The target, with the same rows as X.
		folds: The folds, as row positions.

	Returns:
		One row per fold (index "fold", counted from 1) with the columns "train rmse" and "validation rmse", and
		"epochs" (how many epochs the model ran before it stopped) when the model is an SGD model.
	"""
	scores = cross_validate(
		pipeline, X, y,
		cv=folds,
		scoring='neg_root_mean_squared_error',  # sklearn maximizes scores, so RMSE comes back negated
		return_train_score=True,
		return_estimator=True,
	)
	out = pd.DataFrame(
		{'train rmse': -scores['train_score'], 'validation rmse': -scores['test_score']},
		index=pd.RangeIndex(1, len(folds) + 1, name='fold'),
	)

	models = [final_model(fitted) for fitted in scores['estimator']]
	if hasattr(models[0], 'n_iter_'):
		out['epochs'] = [model.n_iter_ for model in models]

	return out

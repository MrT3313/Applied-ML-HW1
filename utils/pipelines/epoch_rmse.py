import warnings
from typing import Union

import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error

from utils.pipelines.cv_rmse import Folds, final_model


def epoch_rmse(
	*,
	pipeline: TransformedTargetRegressor,
	X: pd.DataFrame,
	y: pd.Series,
	folds: Folds,
	epochs: Union[int, list[int]],
) -> pd.DataFrame:
	"""Training and validation RMSE of an SGD pipeline from build_pipeline after each epoch k, on every fold: a fresh fit
	with max_iter=k and the same seed, which is exactly the first k epochs of the full fit (flat once tol stops it).

	Args:
		pipeline: Pipeline from build_pipeline whose model is an SGDRegressor. It is cloned, not changed.
		X: The attributes.
		y: The target, with the same rows as X.
		folds: The folds, as row positions.
		epochs: The epochs to score on each fold: a count (every epoch from 1 to it), or a list of epoch numbers in
			increasing order. Epoch k costs a fit of k epochs, so a long run is scored on a spaced list (ex log spaced).

	Returns:
		One row per fold and scored epoch (index levels "fold" and "epoch", both counted from 1), with the columns
		"train rmse" and "validation rmse".
	"""
	rows = []
	for fold, (train, validation) in enumerate(folds, start=1):
		X_train, y_train = X.iloc[train], y.iloc[train]
		X_validation, y_validation = X.iloc[validation], y.iloc[validation]

		stopped = False
		for epoch in (range(1, epochs + 1) if isinstance(epochs, int) else epochs):
			if not stopped:
				with warnings.catch_warnings():
					warnings.simplefilter('ignore', ConvergenceWarning)  # max_iter=epoch ends the fit early on purpose
					# make_pipeline names the model's step after its class, so the SGD model is 'sgdregressor'
					fitted = clone(pipeline).set_params(regressor__sgdregressor__max_iter=epoch).fit(X_train, y_train)
				stopped = final_model(fitted).n_iter_ < epoch  # tol stopped it: every longer run is this same model
				scores = (
					mean_squared_error(y_train, fitted.predict(X_train), squared=False),
					mean_squared_error(y_validation, fitted.predict(X_validation), squared=False),
				)
			rows.append({'fold': fold, 'epoch': epoch, 'train rmse': scores[0], 'validation rmse': scores[1]})

	return pd.DataFrame(rows).set_index(['fold', 'epoch'])

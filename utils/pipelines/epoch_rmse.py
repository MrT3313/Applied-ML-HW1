import warnings

import numpy as np
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
	epochs: int,
	method: str = 'warm_start',
) -> pd.DataFrame:
	"""Training and validation RMSE of an SGD pipeline from build_pipeline after each epoch, on every fold, by one of two
	methods.

	'warm_start' is the homework's loop: a copy of the SGD model with max_iter=1 and warm_start=True, fit once per
	epoch, each fit going on from the coefficients the last one left. The loop works on the pipeline's steps, not on
	the whole pipeline: TransformedTargetRegressor fits a fresh copy of its model on every fit (sklearn
	compose/_target.py), which would throw away the coefficients and silently undo warm_start. So on each fold the
	column steps and the target scaler are fit once on the training rows, and only the SGD model is fit in the loop,
	on the columns they give.

	'refit' is the exact curve: epoch k is a fresh fit of a copy of the whole pipeline with max_iter=k and the same
	seed and tol, which is exactly the first k epochs of the full fit. Once tol stops a fold's fit before k epochs,
	every longer run is that same model, so its scores are repeated (a flat line) instead of refit.

	Both exist because the homework's loop is not an exact replay of one long fit. Every fit call restarts the learning
	rate schedule (sklearn sets its step counter t_ back to 1 in fit), so under invscaling the step starts again from
	eta0 each epoch and adaptive never cuts its step (it only cuts after 5 stalled epochs within one fit). Every fit
	call also draws its shuffle from the same seed, so each epoch visits the rows in the first epoch's order. Refit is
	exact for every schedule, at the cost of a fit of k epochs for epoch k.

	Args:
		pipeline: Pipeline from build_pipeline whose model is an SGDRegressor. Its steps are cloned, not changed.
		X: The attributes.
		y: The target, with the same rows as X.
		folds: The folds, as row positions.
		epochs: How many epochs to run on each fold, every one scored.
		method: 'warm_start' (the homework's loop) or 'refit' (the exact curve).

	Returns:
		One row per fold and epoch (index levels "fold" and "epoch", both counted from 1), with the columns
		"train rmse" and "validation rmse".
	"""
	if method not in ('warm_start', 'refit'):
		raise ValueError(f"method must be 'warm_start' or 'refit', not {method!r}")

	rows = []
	for fold, (train, validation) in enumerate(folds, start=1):
		X_train, y_train = X.iloc[train], y.iloc[train]
		X_validation, y_validation = X.iloc[validation], y.iloc[validation]

		if method == 'refit':
			stopped = False
			for epoch in range(1, epochs + 1):
				if not stopped:
					with warnings.catch_warnings():
						warnings.simplefilter('ignore', ConvergenceWarning)  # max_iter=epoch ends the fit early on purpose
						# make_pipeline names the model's step after its class, so the SGD model is 'sgdregressor'
						fitted = clone(pipeline).set_params(regressor__sgdregressor__max_iter=epoch).fit(X_train, y_train)
					stopped = final_model(pipeline=fitted).n_iter_ < epoch  # tol stopped it: every longer run is this same model
					scores = (
						mean_squared_error(y_train, fitted.predict(X_train), squared=False),
						mean_squared_error(y_validation, fitted.predict(X_validation), squared=False),
					)
				rows.append({'fold': fold, 'epoch': epoch, 'train rmse': scores[0], 'validation rmse': scores[1]})
			continue

		# the column steps (imputer, shapes, scaling, expansion), fit on the training rows of the fold once
		columns = clone(pipeline.regressor[:-1]).fit(X_train)
		C_train, C_validation = columns.transform(X_train), columns.transform(X_validation)

		# the target scaler, fit on the training target; the model learns the standardized target
		target = clone(pipeline.transformer).fit(y_train.to_numpy().reshape(-1, 1))
		z_train = target.transform(y_train.to_numpy().reshape(-1, 1)).ravel()

		def in_target_units(*, predictions: np.ndarray) -> np.ndarray:
			return target.inverse_transform(predictions.reshape(-1, 1)).ravel()

		model = clone(pipeline.regressor[-1]).set_params(max_iter=1, warm_start=True)
		for epoch in range(1, epochs + 1):
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', ConvergenceWarning)  # max_iter=1 ends every fit after one epoch on purpose
				model.fit(C_train, z_train)
			rows.append({
				'fold': fold,
				'epoch': epoch,
				'train rmse': mean_squared_error(y_train, in_target_units(predictions=model.predict(C_train)), squared=False),
				'validation rmse': mean_squared_error(y_validation, in_target_units(predictions=model.predict(C_validation)), squared=False),
			})

	return pd.DataFrame(rows).set_index(['fold', 'epoch'])

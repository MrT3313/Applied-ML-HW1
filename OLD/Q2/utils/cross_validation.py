import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit


def rmse_of(actual, predicted):
	"""Root mean squared error: the typical miss, in target units."""
	return np.sqrt(np.mean((actual - predicted) ** 2))


def make_time_folds(*, data: pd.DataFrame, date_col: str, n_splits: int, test_size: int) -> list[tuple[pd.Index, pd.Index]]:
	"""(train_index, val_index) per fold, in time order: every fold trains on the past and validates on what follows.

	Expanding window. The last n_splits * test_size rows are cut into n_splits validation blocks of test_size
	rows. Each fold validates on one block and trains on every row before it, so the training side grows from
	fold to fold and the last block ends at the last row.

	Build the folds once and pass the same folds to every experiment, so every experiment is validated on the
	exact same rows.

	Args:
		data: Training data, sorted by date_col. Its index identifies the rows in the returned folds and must be unique.
		date_col: Column the rows are ordered by.
		n_splits: Number of folds.
		test_size: Rows in each validation block.

	Returns:
		n_splits folds, earliest validation block first.
	"""
	# Folds are stored as index values (select with .loc), so they keep pointing at the same days when derived
	# columns are added to data later.
	assert data.index.is_unique, "data.index must be unique"
	# A fold is only "train on the past" when the rows are in date order.
	assert data[date_col].is_monotonic_increasing, f"data must be sorted by {date_col}"

	folds = []
	# TimeSeriesSplit returns POSITIONS (0th, 1st, ...), not index values.
	# ex 584 rows, 6 folds of 31: train 0..397 / val 398..428, then train 0..428 / val 429..459, ...
	for train_pos, val_pos in TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(data):
		# positions -> index values
		folds.append((data.index[train_pos], data.index[val_pos]))

	return folds


def test_train_ratio(*, folds: list[tuple[pd.Index, pd.Index]]) -> float:
	"""Mean validation rows / training rows of the folds, the n_test / n_train term of standard_error.

	The training side of time-ordered folds grows from fold to fold, so the ratio differs per fold and the mean
	is used.

	Args:
		folds: Output of make_time_folds.

	Returns:
		The mean ratio, ex about 0.066 for 6 folds of 31 rows on 584 rows.
	"""
	return float(np.mean([len(val_index) / len(train_index) for train_index, val_index in folds]))


def cv_model(
	*,
	data: pd.DataFrame,
	columns: list[str],
	estimator: BaseEstimator,
	folds: list[tuple[pd.Index, pd.Index]],
	target: str,
) -> tuple[np.ndarray, np.ndarray]:
	"""Validation and training RMSE of `estimator` on each fold.

	Args:
		data: The same DataFrame the folds were made from (extra derived columns are fine).
		columns: Columns the estimator is fit on.
		estimator: Estimator or pipeline. A fresh clone is fit on each fold.
		folds: Output of make_time_folds.
		target: Column the estimator predicts.

	Returns:
		(val_scores, train_scores), one RMSE per fold each.
	"""
	val_scores, train_scores = [], []
	for train_index, val_index in folds:
		# .loc selects rows by INDEX (the folds hold index values)
		train, val = data.loc[train_index], data.loc[val_index]
		fitted = clone(estimator).fit(train[columns], train[target])
		val_scores.append(rmse_of(val[target], fitted.predict(val[columns])))
		train_scores.append(rmse_of(train[target], fitted.predict(train[columns])))
	return np.array(val_scores), np.array(train_scores)

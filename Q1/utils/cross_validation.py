import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from Q1.consts import SEEDS, TARGET


def rmse_of(actual, predicted):
	"""Root mean squared error: the typical miss, in target units."""
	return np.sqrt(np.mean((actual - predicted) ** 2))


def make_folds(*, data, feature_columns, seeds=SEEDS, n_splits=4):
	"""(train_index, val_index) per fold. Validation holds complete rows only; incomplete rows always train.

	Build the folds once and pass the same folds to every experiment, so every experiment is validated on the
	exact same rows.

	Args:
		data: Training data. Its index identifies the rows in the returned folds and must be unique.
		feature_columns: Every candidate feature column. A row is complete when all of them are present.
		seeds: One KFold random_state per repeat.
		n_splits: Number of folds per seed.

	Returns:
		len(seeds) * n_splits folds, seed by seed.
	"""
	# Two ways to point at a row:
	#   index:    the column of row names pandas prints down the left side (data.index). read_csv numbers the
	#             rows 0, 1, 2, ... and a row KEEPS its index value when other rows are removed
	#             (ex after dropna, the index might be 0, 2, 3, 5). Select with .loc.
	#   position: "the nth row right now", counting from 0. It shifts when rows are removed. Select with .iloc.
	# Folds are stored as index values so they always point at the same countries/years, whatever subset is
	# used later.
	#
	# Running example used in the comments below: 6 rows with index 0..5, rows 1 and 4 have a missing feature.

	# Two rows sharing an index value would be ambiguous ("which row 3?"), so the index must be unique.
	assert data.index.is_unique, "data.index must be unique"

	# notna() -> True/False for every cell; all(axis=1) -> True for a row only if every feature is present.
	# data.index[mask] keeps the index values of the True rows.    ex complete_index = [0, 2, 3, 5]
	complete_index = data.index[data[feature_columns].notna().all(axis=1)]

	folds = []
	for seed in seeds:
		# KFold only counts the complete rows, so only complete rows can end up in validation.
		# It returns POSITIONS within complete_index (0th, 1st, ...), not index values. The training positions
		# are ignored (_) because the training side is rebuilt below to also include the incomplete rows.
		for _, val_pos in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(complete_index):
			# positions -> index values                              ex val_pos = [1, 3] -> val_index = [2, 5]
			val_index = complete_index[val_pos]

			# training = every row of data that is not in validation, complete or not.
			# ex data.index = [0, 1, 2, 3, 4, 5] minus [2, 5] -> train_index = [0, 1, 3, 4]
			# (rows 1 and 4 are incomplete: cv_model drops them or lets an imputer use them)
			train_index = data.index.difference(val_index)
			folds.append((train_index, val_index))

	# len(seeds) * n_splits pairs. Every complete row is in validation exactly once per seed.
	return folds


def cv_model(*,
						 data,
						 columns,
						 estimator,
						 folds,
						 drop_incomplete=True
):
	"""Validation and training RMSE of `estimator` on each fold.

	Args:
		data: The same DataFrame the folds were made from (extra derived columns are fine).
		columns: Columns the estimator is fit on.
		estimator: Estimator or pipeline. A fresh clone is fit on each fold.
		folds: Output of make_folds.
		drop_incomplete: True trains on complete rows only. False also trains on incomplete rows, so the
			estimator must handle missing values (ex a pipeline with an imputer).

	Returns:
		(val_scores, train_scores), one RMSE per fold each.
	"""
	val_scores, train_scores = [], []
	for train_index, val_index in folds:
		# .loc selects rows by INDEX (the folds hold index values). .iloc would select by position, which is
		# what the step-by-step notebook used because KFold there split `data` directly and returned positions.
		train, val = data.loc[train_index], data.loc[val_index]
		if drop_incomplete:
			train = train.dropna()  # complete rows only, the same set in every experiment
		fitted = clone(estimator).fit(train[columns], train[TARGET])
		val_scores.append(rmse_of(val[TARGET], fitted.predict(val[columns])))
		train_scores.append(rmse_of(train[TARGET], fitted.predict(train[columns])))
	return np.array(val_scores), np.array(train_scores)

import pandas as pd

from Q1.utils.ML.training.complete_row_folds import Folds, complete_row_folds


def repeated_folds(
	X: pd.DataFrame,
	*,
	seeds: list[int],
	n_splits: int = 4,
	add_incomplete_to_train: bool = False,
) -> Folds:
	"""Repeat complete_row_folds once per seed and join the folds into one flat list.

	Args:
		X: Feature data. Positions in the returned folds are relative to X.
		seeds: One KFold random_state per repeat.
		n_splits: Number of folds per seed.
		add_incomplete_to_train: Passed to complete_row_folds.

	Returns:
		len(seeds) * n_splits (train_positions, test_positions) pairs, seed by seed, usable directly as cv= in cross_validate.
	"""
	return [
		fold
		for seed in seeds
		for fold in complete_row_folds(X, seed=seed, n_splits=n_splits, add_incomplete_to_train=add_incomplete_to_train)
	]

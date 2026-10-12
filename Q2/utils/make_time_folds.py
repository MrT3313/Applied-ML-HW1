import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from utils.pipelines.make_folds import CVFolds


def make_time_folds(df: pd.DataFrame, *, date_col: str, n_splits: int, validation_size: int) -> CVFolds:
	"""Split rows into time-ordered folds: every fold trains on the past and validates on the days that follow.

	Expanding window. The last n_splits * validation_size rows are cut into n_splits validation blocks of
	validation_size rows. Each fold validates on one block and trains on every row before it, so the training
	side grows from fold to fold and the last block ends at the last row. Nothing is shuffled, so there is no
	seed.

	Args:
		df: Rows to split, sorted by date_col. Its index labels identify the rows in the returned folds.
		date_col: Column the rows are ordered by.
		n_splits: Number of folds.
		validation_size: Rows in each validation block.

	Returns:
		n_splits folds, earliest validation block first, in the same CVFolds format as make_folds.
	"""
	assert df.index.is_unique, "df.index must be unique"
	# a fold only trains on the past when the rows are in date order
	assert df[date_col].is_monotonic_increasing, f"df must be sorted by {date_col}"

	# ex 584 rows, 4 folds of 46: train 0..399 / validation 400..445, then train 0..445 / validation 446..491, ...
	splitter = TimeSeriesSplit(n_splits=n_splits, test_size=validation_size)
	folds = [(df.index[train], df.index[validation]) for train, validation in splitter.split(df)]
	return CVFolds(folds, seeds=[], n_splits=n_splits)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# one (train, validation) pair of positional row indices per fold, the format cross_validate accepts as cv=
Folds = list[tuple[np.ndarray, np.ndarray]]


def complete_row_folds(
	X: pd.DataFrame,
	*,
	seed: int,
	n_splits: int = 4,
	add_incomplete_to_train: bool,
) -> Folds:
	"""Split the rows of X with no missing values into seeded KFold folds.

	The seeded KFold runs on the complete rows only, so the validation folds are the same in every experiment,
	whether or not it keeps the incomplete rows. Incomplete rows never land in a validation fold.

	Args:
		X: Feature data. Positions in the returned folds are relative to X, so pass the same X to cross_validate.
		seed: random_state of the shuffled KFold.
		n_splits: Number of folds.
		add_incomplete_to_train: Add every incomplete row to each training fold (for pipelines with an imputer).

	Returns:
		One (train_positions, test_positions) pair per fold.
	"""
	complete = X.notna().all(axis=1).to_numpy()
	complete_positions = np.flatnonzero(complete)
	incomplete_positions = np.flatnonzero(~complete)

	folds = []
	for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(complete_positions):
		# map positions within the complete rows back to positions within X
		train, test = complete_positions[train], complete_positions[test]
		if add_incomplete_to_train:
			train = np.sort(np.concatenate([train, incomplete_positions]))
		folds.append((train, test))
	return folds

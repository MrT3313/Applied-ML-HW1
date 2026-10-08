import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# one (train, validation) pair of positional row indices per fold, the format cross_validate accepts as cv=
Folds = list[tuple[np.ndarray, np.ndarray]]

def complete_row_folds(X: pd.DataFrame,
					   *,
					   seed: int,
					   n_splits: int = 4,
					   add_incomplete_to_train: bool
) -> Folds:
	# the seeded KFold runs on the complete rows only (no missing features), so the validation folds are
	# the same in every experiment, whether or not it keeps the incomplete rows.
	# with add_incomplete_to_train, every incomplete row is added to each training fold (for pipelines with
	# an imputer); incomplete rows never land in a validation fold.
	# positions are relative to X, so pass the same X to run_experiment
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

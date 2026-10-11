import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class CVFolds:
	"""Cross validation folds stored as index labels, so the same folds can be applied to any subset of the rows.

	Args:
		folds: One (train_labels, validation_labels) pair per fold, in fold order (seed by seed, then fold).
		seeds: KFold random_state of each repeat.
		n_splits: Number of folds per seed.
	"""

	def __init__(
		self,
		folds: list[tuple[pd.Index, pd.Index]],
		*,
		seeds: list[int],
		n_splits: int,
	):
		self.folds = folds
		self.seeds = list(seeds)
		self.n_splits = n_splits

	def __len__(self) -> int:
		return len(self.folds)

	def positions_for(self, X: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
		"""Convert the folds to positions within X, the format cross_validate accepts as cv=.

		Every validation label must be in X. Training labels not in X (ex incomplete rows dropped from X) are
		left out, so a pipeline trains only on the rows it was given.

		Args:
			X: Feature data whose index holds the labels of the folds. Its index must be unique.

		Returns:
			One (train_positions, validation_positions) pair per fold, each sorted ascending.
		"""
		assert X.index.is_unique, "X.index must be unique"

		positions = []
		for train_labels, validation_labels in self.folds:
			validation = X.index.get_indexer(validation_labels)
			n_missing = int(np.sum(validation < 0))
			if n_missing:
				raise ValueError(f"{n_missing} validation labels are not in X.index")
			train = X.index.get_indexer(train_labels)
			# keep only the training rows present in X
			train = train[train >= 0]
			positions.append((np.sort(train), np.sort(validation)))
		return positions


def make_folds(
	X: pd.DataFrame,
	*,
	seeds: list[int],
	n_splits: int = 4,
) -> CVFolds:
	"""Split the complete rows of X (no missing values) into seeded KFold folds, repeated once per seed.

	Validation folds hold complete rows only, so every pipeline is validated on the exact same rows. The training
	side of each fold holds every other row of X, complete or not. Through positions_for, pipelines that drop
	incomplete rows train on complete rows only, and pipelines that impute also train on the incomplete rows,
	automatically.

	Args:
		X: Feature data. Its index labels identify the rows in the returned folds.
		seeds: One KFold random_state per repeat.
		n_splits: Number of folds per seed.

	Returns:
		len(seeds) * n_splits folds, seed by seed.
	"""
	complete = X.notna().all(axis=1).to_numpy()
	complete_positions = np.flatnonzero(complete)

	folds = []
	for seed in seeds:
		for _, test in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(complete_positions):
			# map positions within the complete rows back to positions within X
			in_validation = np.zeros(len(X), dtype=bool)
			in_validation[complete_positions[test]] = True
			folds.append((X.index[~in_validation], X.index[in_validation]))
	return CVFolds(folds, seeds=seeds, n_splits=n_splits)

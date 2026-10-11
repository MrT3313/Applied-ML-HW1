import numpy as np


def standard_error(*, diff, n_splits=4):
	"""Corrected standard error of the mean per-fold difference between two models (Nadeau and Bengio).

	The folds share most of their training rows, so their scores move together and the plain sd / sqrt(n)
	understates the noise. The correction adds n_test / n_train, which is 1 / (n_splits - 1), ex 1/3 for 4 folds.

	Args:
		diff: Per-fold differences, ex compare val RMSE minus baseline val RMSE, both scored on the same folds.
		n_splits: Folds per seed the scores were made with.

	Returns:
		The corrected SE, in the same units as diff.
	"""
	diff = np.asarray(diff)
	return np.sqrt(np.var(diff, ddof=1) * (1 / len(diff) + 1 / (n_splits - 1)))

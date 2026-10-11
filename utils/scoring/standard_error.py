import numpy as np


def standard_error(*, diff, n_splits=4, test_train_ratio: float | None = None):
	"""Corrected standard error of the mean per-fold difference between two models (Nadeau and Bengio).

	The folds share most of their training rows, so their scores move together and the plain sd / sqrt(n)
	understates the noise. The correction adds n_test / n_train. For KFold that is 1 / (n_splits - 1), ex 1/3 for
	4 folds. Folds made another way (ex time-ordered folds) pass their own ratio as test_train_ratio.

	Args:
		diff: Per-fold differences, ex compare val RMSE minus baseline val RMSE, both scored on the same folds.
		n_splits: Folds per seed the scores were made with. Only used when test_train_ratio is None.
		test_train_ratio: n_test / n_train of the folds. None uses the KFold value, 1 / (n_splits - 1).

	Returns:
		The corrected SE, in the same units as diff.
	"""
	diff = np.asarray(diff)
	if test_train_ratio is None:
		test_train_ratio = 1 / (n_splits - 1)
	return np.sqrt(np.var(diff, ddof=1) * (1 / len(diff) + test_train_ratio))

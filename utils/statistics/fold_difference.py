from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.pipelines.cv_rmse import Folds


# frozen=True makes a result read-only once it is built; kw_only=True means it can only be built with named values.
@dataclass(frozen=True, kw_only=True)
class FoldDifference:
	"""How a candidate's validation RMSE differs from a baseline's, fold by fold, on the same folds.

	Attributes:
		diff: Candidate's validation RMSE minus the baseline's on each fold. Below 0, the candidate is better there.
		se: Standard error of the mean of diff, widened because the folds share training rows (see fold_difference).
	"""
	diff: np.ndarray
	se: float

	@property
	def mean(self) -> float:
		"""The mean change in validation RMSE (Δrmse)."""
		return float(self.diff.mean())

	@property
	def better(self) -> int:
		"""Number of folds where the candidate has the lower validation RMSE."""
		return int((self.diff < 0).sum())

	@property
	def verdict(self) -> str:
		"""The 1 SE rule: 'candidate' when Δrmse is below -1 SE, 'worse' when above +1 SE, otherwise 'within noise'."""
		if self.mean < -self.se:
			return 'candidate'
		if self.mean > self.se:
			return 'worse'
		return 'within noise'


def fold_difference(*, baseline: pd.DataFrame, candidate: pd.DataFrame, folds: Folds) -> FoldDifference:
	"""Compare a candidate's cross validation result with a baseline's, fold by fold.

	The standard error is the Nadeau and Bengio correction: the plain sd / sqrt(folds) is too small, because the
	folds share most of their training rows and so their results move together. The correction adds the share
	validation rows / training rows (the mean over the folds) inside the square root.

	Args:
		baseline: The baseline's result from PIPELINES.cv_rmse.
		candidate: The candidate's result from PIPELINES.cv_rmse, on the same folds.
		folds: The folds both were scored on, as row positions.

	Returns:
		The per-fold differences and their corrected standard error.
	"""
	diff = (candidate['validation rmse'] - baseline['validation rmse']).to_numpy()
	validation_share = np.mean([len(validation) / len(train) for train, validation in folds])
	se = np.sqrt(np.var(diff, ddof=1) * (1 / len(diff) + validation_share))
	return FoldDifference(diff=diff, se=float(se))

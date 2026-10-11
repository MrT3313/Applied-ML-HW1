import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ThresholdKind = Literal['step', 'hinge']
THRESHOLD_KINDS: tuple[ThresholdKind, ...] = ('step', 'hinge')


@dataclass(frozen=True)
class ThresholdResult:
	"""A step or hinge fit of target on feature, split at one threshold.

	Attributes:
		kind: 'step' predicts each side with its own mean.
			'hinge' fits y = a + b·x + c·max(0, x - t), two line segments joined at t.
		threshold: The split point t. Rows with feature < t are the low side, matching box_plot.
		coefficients: (low_mean, high_mean) for 'step', (a, b, c) for 'hinge'.
		r_squared: R² of this fit.
		linear_r_squared: R² of a plain linear fit on the same rows, for comparison.
	"""
	kind: ThresholdKind
	threshold: float
	coefficients: tuple[float, ...]
	r_squared: float
	linear_r_squared: float

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Evaluate the fit at x."""
		x = np.asarray(x, dtype=float)
		if self.kind == 'step':
			low_mean, high_mean = self.coefficients
			return np.where(x < self.threshold, low_mean, high_mean)
		a, b, c = self.coefficients
		return a + b * x + c * np.maximum(0, x - self.threshold)


def _fit(kind: ThresholdKind, x: np.ndarray, y: np.ndarray, threshold: float) -> tuple[tuple[float, ...], float]:
	"""Fit one kind at a fixed threshold. Returns its coefficients and sum of squared errors."""
	if kind == 'step':
		low = x < threshold
		coefficients = (float(y[low].mean()), float(y[~low].mean()))
	elif kind == 'hinge':
		design = np.column_stack([np.ones_like(x), x, np.maximum(0, x - threshold)])
		coefficients = tuple(float(v) for v in np.linalg.lstsq(design, y, rcond=None)[0])
	else:
		raise ValueError(f'Unknown kind: {kind}')

	result = ThresholdResult(kind, threshold, coefficients, np.nan, np.nan)
	sse = float(np.sum((y - result.predict(x)) ** 2))
	return coefficients, sse


def _xy(df: pd.DataFrame, feature: str, target: str) -> tuple[np.ndarray, np.ndarray]:
	data = df[[feature, target]].dropna()
	return data[feature].to_numpy(dtype=float), data[target].to_numpy(dtype=float)


def _result(kind: ThresholdKind, x: np.ndarray, y: np.ndarray, threshold: float, coefficients: tuple[float, ...], sse: float) -> ThresholdResult:
	sst = float(np.sum((y - y.mean()) ** 2))
	linear_r_squared = float(np.corrcoef(x, y)[0, 1] ** 2)
	return ThresholdResult(kind, float(threshold), coefficients, 1 - sse / sst, linear_r_squared)


def fit_threshold(*, df: pd.DataFrame, feature: str, target: str, kind: ThresholdKind, threshold: float) -> ThresholdResult:
	"""Fit a step or hinge of target on feature at a threshold you choose.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Column the split is made on.
		target: Column being predicted.
		kind: 'step' or 'hinge' (see ThresholdResult).
		threshold: Split point. Rows with feature < threshold are the low side.

	Returns:
		The fit at that threshold.
	"""
	x, y = _xy(df, feature, target)
	coefficients, sse = _fit(kind, x, y, threshold)
	return _result(kind, x, y, threshold, coefficients, sse)


def find_threshold(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	kind: ThresholdKind,
	min_group_frac: float = 0.05,
) -> ThresholdResult:
	"""Search every distinct feature value for the threshold whose step or hinge fit has the lowest squared error.

	The search sees the same rows it is scored on, so the best R² is optimistic. Run it on training data only.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Column the split is made on.
		target: Column being predicted.
		kind: 'step' or 'hinge' (see ThresholdResult).
		min_group_frac: Smallest share of rows allowed on each side of the split, so a split
			near the edge cannot win by isolating a few outliers.

	Returns:
		The best fit found.
	"""
	x, y = _xy(df, feature, target)
	min_rows = math.ceil(min_group_frac * len(x))

	# a candidate t puts every row with x < t on the low side
	candidates = np.unique(x)
	n_low = np.searchsorted(np.sort(x), candidates, side='left')
	candidates = candidates[(n_low >= min_rows) & (len(x) - n_low >= min_rows)]
	if len(candidates) == 0:
		raise ValueError(f'No threshold leaves {min_group_frac:.0%} of rows on each side')

	best = None
	for threshold in candidates:
		coefficients, sse = _fit(kind, x, y, threshold)
		if best is None or sse < best[2]:
			best = (threshold, coefficients, sse)

	threshold, coefficients, sse = best
	return _result(kind, x, y, threshold, coefficients, sse)

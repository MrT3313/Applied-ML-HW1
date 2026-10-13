"""Find the feature value where the relationship between a feature and the target changes.

The big picture:
	A straight line assumes the target moves at the same rate across the whole range of a feature.
	Often it does not: the target may jump at some value, or rise quickly and then level off.
	This module tests two simple "bent" shapes that can capture that, each built around one split
	point on the feature (the threshold, t):

	- step:  one flat level below t and another flat level at or above t.
	- hinge: one straight line below t and a second straight line above t, joined at t.

	fit_threshold fits one of those shapes at a threshold you pick.
	find_threshold tries every possible threshold and keeps the one that fits best.

How a fit is graded:
	Every fit is graded by its squared error: for each row, take (actual - predicted) and square it,
	then add the rows up (SSE, "sum of squared errors"). Lower is better. R² rescales that into the
	share of the target's variation the fit explains, where 1 is perfect and 0 is no better than
	always predicting the average.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# the two shapes this module knows how to fit
ThresholdKind = Literal['step', 'hinge']
THRESHOLD_KINDS: tuple[ThresholdKind, ...] = ('step', 'hinge')


# frozen=True makes a result read-only once it is built, so a plot cannot change it by accident.
# kw_only=True means it can only be built with named values (kind=..., threshold=...), never by position.
@dataclass(frozen=True, kw_only=True)
class ThresholdResult:
	"""A step or hinge fit of target on feature, split at one threshold.

	Attributes:
		kind: 'step' predicts each side with its own mean.
			'hinge' fits y = intercept + slope·x + slope_change·max(0, x - t), two line segments joined at t.
		threshold: The split point t. Rows with feature < t are the low side, matching box_plot.
		coefficients: (low_mean, high_mean) for 'step', (intercept, slope, slope_change) for 'hinge'.
		r_squared: R² of this fit on the rows it was fitted to. Optimistic when the threshold was searched.
	"""
	kind: ThresholdKind
	threshold: float
	coefficients: tuple[float, ...]
	r_squared: float

	def predict(self, *, x: np.ndarray) -> np.ndarray:
		"""Evaluate the fit at x."""
		return _predict(kind=self.kind, threshold=self.threshold, coefficients=self.coefficients, x=x)

def _xy(*, df: pd.DataFrame, feature: str, target: str) -> tuple[np.ndarray, np.ndarray]:
	"""Pull the feature (x) and target (y) out of the DataFrame as plain number arrays."""
	# a row missing either value cannot be fitted or graded, so it is dropped here once for everything below
	data = df[[feature, target]].dropna()
	return data[feature].to_numpy(dtype=float), data[target].to_numpy(dtype=float)

def _search(*, kind: ThresholdKind, x: np.ndarray, y: np.ndarray, min_group_frac: float) -> tuple[float, tuple[float, ...], float]:
	"""Try every distinct x as the threshold. Returns the one with the lowest squared error, its coefficients and that error."""
	# the fewest rows allowed on either side of a split (5% of the rows by default)
	min_rows = math.ceil(min_group_frac * len(x))

	# The candidates are the values that actually occur in the data.
	# a candidate t puts every row with x < t on the low side
	#
	# Example with x = [0.77, 0.85, 0.90]:
	#   t = 0.85                     -> low side {0.77},       high side {0.85, 0.90}
	#   t = 0.86, 0.875, 0.89, 0.90  -> low side {0.77, 0.85}, high side {0.90}
	#
	# For a step, every t inside the gap (above 0.85, up to 0.90) groups the rows exactly like t = 0.90
	# does, so it gives the same two averages and the same error. The training rows cannot tell those
	# thresholds apart, so trying 0.90 stands in for the whole gap and the returned threshold is the
	# gap's upper end.
	#
	# For a hinge it is an approximation: the error does change as t moves inside a gap, so the best
	# bend point can fall between two values, where this search never looks.
	# Example with x = [1, 2, 3, 4, 5, 6] and y = [1, 2, 3, 5, 8, 11] (the slope changes at 3.5):
	#   t = 3    -> squared error 0.316
	#   t = 3.5  -> squared error 0.000 (a perfect fit, but 3.5 is not in the data so it is never tried)
	#   t = 4    -> squared error 0.316
	# So a hinge threshold is only as precise as the spacing of the values around it.
	candidates = np.unique(x)

	# for each candidate, count how many rows fall below it (searchsorted on the sorted values does
	# that count for all candidates at once)
	n_low = np.searchsorted(np.sort(x), candidates, side='left')

	# Drop candidates that leave too few rows on one side. Without this, a split at the very edge
	# could win just by giving two or three unusual rows their own perfect little group.
	candidates = candidates[(n_low >= min_rows) & (len(x) - n_low >= min_rows)]
	if len(candidates) == 0:
		raise ValueError(f'No threshold leaves {min_group_frac:.0%} of rows on each side')

	# brute force: fit the shape at every remaining candidate and remember the one with the lowest error
	best = None
	for threshold in candidates:
		coefficients, sse = _fit(kind=kind, x=x, y=y, threshold=threshold)
		# best[2] is the lowest squared error seen so far
		if best is None or sse < best[2]:
			best = (float(threshold), coefficients, sse)

	return best

def _fit(*, kind: ThresholdKind, x: np.ndarray, y: np.ndarray, threshold: float) -> tuple[tuple[float, ...], float]:
	"""Fit one kind at a fixed threshold. Returns its coefficients and sum of squared errors."""
	if kind == 'step':
		# `low` is True for each row below the threshold, and `~low` flips it to mean "at or above".
		# The best flat level for a group of rows is that group's average (the average is the single
		# number with the smallest squared error), so the fit is just one mean per side.
		low = x < threshold
		coefficients = (float(y[low].mean()), float(y[~low].mean()))
	elif kind == 'hinge':
		# The hinge is not a special kind of fit. It is an ordinary linear regression, and the bend
		# comes from the inputs it is given. A regression predicts by multiplying each input column by
		# its own number and adding the results:
		#   prediction = intercept·(column 1) + slope·(column 2) + slope_change·(column 3)
		#
		# Compared with a straight line y = mx + b: intercept is the b, slope is the m, and slope_change
		# is the one extra number a hinge needs, which is how much the slope changes at the threshold.
		#
		# The three columns built for each row are:
		#   column 1: always 1
		#       intercept·1 is just intercept, the same for every row. This gives the line its starting height.
		#   column 2: x itself
		#       slope·x is the ordinary slope. It applies to every row, on both sides of the threshold.
		#   column 3: max(0, x - threshold), how far x is past the threshold (0 if it has not reached it)
		#       x - threshold is negative for a row below the threshold, and max(0, ...) replaces a
		#       negative with 0. With threshold = 3.5:  x = 3 -> max(0, -0.5) = 0,  x = 5 -> max(0, 1.5) = 1.5.
		#       Below the threshold this column is 0, and slope_change·0 = 0, so slope_change adds nothing
		#       to those rows no matter what it is. Above the threshold the column goes up by 1 each time
		#       x goes up by 1, so the prediction goes up by an extra slope_change on top of the slope it
		#       already gets from column 2.
		#       It works like overtime pay: slope is the rate for every hour, slope_change is the bonus
		#       rate, and this column is the number of hours worked past the cutoff.
		# The result is a slope of `slope` below the threshold and `slope + slope_change` above it.
		# slope_change = 0 means no bend at all, positive means steeper, negative means flatter.
		# At exactly the threshold column 3 is 0, so the two segments meet with no jump.
		#
		# Example with x = [1, 2, 3, 4, 5, 6] and threshold = 3.5:
		#   x = 3 -> columns (1, 3, 0)     below the threshold, column 3 is off
		#   x = 5 -> columns (1, 5, 1.5)   above it, column 3 is 5 - 3.5
		# If y = [1, 2, 3, 5, 8, 11], the fit below returns intercept = 0, slope = 1, slope_change = 2, and
		# the x = 5 row is predicted as 0·1 + 1·5 + 2·1.5 = 8. The line rises by 1 per unit below 3.5 and
		# by 3 (1 + 2) above it.

		# column_stack glues the three columns side by side into one table with a row per data row
		design = np.column_stack([np.ones_like(x), x, np.maximum(0, x - threshold)])

		# lstsq ("least squares") takes that table and the real target values and finds the intercept,
		# slope and slope_change that make the squared error as small as possible. It returns several
		# things, and [0] is that three-number answer, in the same order as the columns. rcond=None tells
		# it to use its default numerical settings. The tuple(float(...)) wrapper turns the answer into
		# three plain Python numbers.
		coefficients = tuple(float(v) for v in np.linalg.lstsq(design, y, rcond=None)[0])
	else:
		raise ValueError(f'Unknown kind: {kind}')

	# grade the fit: predict every row, square each miss, and add them up
	sse = float(np.sum((y - _predict(kind=kind, threshold=threshold, coefficients=coefficients, x=x)) ** 2))
	return coefficients, sse

def _predict(*, kind: ThresholdKind, threshold: float, coefficients: tuple[float, ...], x: np.ndarray) -> np.ndarray:
	"""Turn feature values into predicted target values, using a shape that has already been fitted."""
	x = np.asarray(x, dtype=float)

	if kind == 'step':
		# a step has only two possible answers: the low side's mean or the high side's mean.
		# np.where picks low_mean for every row below the threshold and high_mean for the rest.
		low_mean, high_mean = coefficients
		return np.where(x < threshold, low_mean, high_mean)

	# hinge: y = intercept + slope·x + slope_change·max(0, x - t)
	#   max(0, x - t) is 0 for rows below t, so there the prediction is just the line intercept + slope·x.
	#   Above t it grows by 1 for every unit of x, so the slope_change term switches on and the slope
	#   becomes slope + slope_change.
	#   At exactly x = t the extra term is 0, so the two segments meet with no jump.
	intercept, slope, slope_change = coefficients
	return intercept + slope * x + slope_change * np.maximum(0, x - threshold)

def _result(*, kind: ThresholdKind, y: np.ndarray, threshold: float, coefficients: tuple[float, ...], sse: float) -> ThresholdResult:
	"""Convert the raw squared error into an R² and package everything as a ThresholdResult."""
	# SST ("total sum of squares") is the squared error of the laziest possible model: always predict
	# the target's average. It is the baseline every fit is measured against.
	sst = float(np.sum((y - y.mean()) ** 2))

	# R² = 1 - SSE / SST: the share of the baseline's error that the fit removed.
	return ThresholdResult(kind=kind, threshold=float(threshold), coefficients=coefficients, r_squared=1 - sse / sst)


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
	x, y = _xy(df=df, feature=feature, target=target)

	# Search all the rows for the best threshold. The search tried many thresholds and kept the winner,
	# and `sse` grades that winner on the same rows it was picked to fit. Some of the good score is the
	# search latching onto quirks of these particular rows, so the R² built from `sse` is a little too
	# high ("optimistic"). The threshold itself is still the best one for these rows.
	threshold, coefficients, sse = _search(kind=kind, x=x, y=y, min_group_frac=min_group_frac)
	return _result(kind=kind, y=y, threshold=threshold, coefficients=coefficients, sse=sse)
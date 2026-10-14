"""Fit many candidate shapes of one feature to the target, to find the ones that beat a straight line.

The big picture:
	A straight line assumes the target moves at the same rate across the whole range of a feature. The LOWESS curve
	on the scatter plot follows the local average of the target without assuming any shape, so it hints at the shape
	the relationship really has. Every candidate here is one standard way of rebuilding the feature as the input
	columns of an ordinary linear regression, so the fitted line can follow a bend. A candidate is worth carrying
	forward when it beats the straight line's R² and gets close to the LOWESS curve.

The three families:
	- re-expression: replace the feature with one new column, ex log(x). The fit is a straight line on the new
	  scale, which is a curve on the original one.
	- piecewise linear: straight segments joined at a threshold, ex a hinge, or a cap that goes flat past it.
	- bins: cut the feature into ranges and predict each range with its own level, a staircase.

	Powers of the feature (x², x³) and other polynomial-like curves are left out on purpose: adding powers of the
	features is what polynomial regression does, so they belong to that comparison, not to this one.

How a candidate is graded:
	R² of the fit on the rows it was fitted to: the share of the target's variation the fit explains, where 1 is
	perfect and 0 is no better than always predicting the average. More columns always fit a little better, so
	every candidate also reports how many columns it uses.

Searched settings:
	Some candidates have a setting that does not change their number of columns, ex the threshold of a hinge or the
	shift and direction of a log. Every value in that setting's grid is tried and the one with the highest R² is kept, the same
	way find_threshold.py searches a threshold. The search sees the same rows it is scored on, so the kept R² is a
	little optimistic.
	The number of bins is fixed instead: more bins means more columns, which almost always raises R² on these rows,
	so a search would always pick the largest number tried.

Notes for reading the results:
	- reciprocal, log and sqrt are the Box-Cox transform at the powers -1, 0 and 0.5. A straight line fits
	  (x^p - 1) / p exactly as well as x^p (one is a rescaled copy of the other), so Box-Cox at those powers is not
	  repeated as its own candidate.
	- reciprocal, log and sqrt each try both directions: as it is (steep at the low end, flat at the high end) and
	  reflected (the mirror image). Yeo-Johnson needs no reflected version: flipping the sign of the feature gives
	  exactly the fit of Yeo-Johnson at the power 2 - λ, so its powers above 1 already cover the mirror image.
"""

import math
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from statsmodels.nonparametric.smoothers_lowess import lowess as fit_lowess

Family = Literal['re-expression', 'piecewise linear', 'bins']
FAMILIES: tuple[Family, ...] = get_args(Family)

# Searched settings (each value is tried, the best R² is kept)
MIN_GROUP_FRAC = 0.05  # hinge, cap, floor and step thresholds leave at least 5% of rows on each side, as in find_threshold.py
SHIFT_STEPS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)  # how far above 0 the bending end lands, as a multiple of the range
WINSORIZE_PERCENTS = range(1, 46)  # clip the lowest p% and the highest p% of values, leaving at least the middle 10% as it is
# Yeo-Johnson powers: 1 leaves the feature as it is, below 1 squeezes its high end (like log), above 1 squeezes its low
# end. The range is wide because features between 0 and 1 need large powers before the curve bends much (Humidity's
# best power is about 24).
YEO_JOHNSON_LAMBDAS = np.arange(-10, 30.01, 0.5)

# Fixed setting (it adds columns, so it is not searched)
N_BINS = 5


# frozen=True makes a result read-only once it is built, so a plot cannot change it by accident.
# kw_only=True means it can only be built with named values (name=..., family=...), never by position.
@dataclass(frozen=True, kw_only=True)
class ShapeFit:
	"""One candidate shape of a feature, fitted to the target.

	Attributes:
		name: The candidate, ex 'log' or 'hinge'.
		family: The family it belongs to (see the module docstring).
		setting: The setting the fit used, ex 't = 0.58' for a searched threshold or '5 bins' for a fixed one, or ''
			when it has none.
		columns: Number of input columns the fit uses, not counting the intercept.
		r_squared: R² on the rows it was fitted to. Optimistic when a setting was searched.
		model: The fitted pipeline, which takes the feature and predicts the target.
	"""
	name: str
	family: Family
	setting: str
	columns: int
	r_squared: float
	model: Pipeline

	@property
	def label(self) -> str:
		"""The name, followed by the setting in parentheses when the shape has one."""
		return f'{self.name} ({self.setting})' if self.setting else self.name

	def predict(self, *, x: np.ndarray) -> np.ndarray:
		"""Evaluate the fit at x."""
		return self.model.predict(np.asarray(x, dtype=float).reshape(-1, 1))


# The transforms below each take the feature as one column (a 2D array with a single column, which is what
# FunctionTransformer passes in) and return the column or columns the linear regression is fitted on.

def _power(x: np.ndarray, *, power: float, shift: float, reflect: bool) -> np.ndarray:
	"""(x + shift), or (shift - x) when reflected, raised to the power, where power 0 means the log."""
	shifted = (-x if reflect else x) + shift
	return np.log(shifted) if power == 0 else shifted ** power

def _yeo_johnson(x: np.ndarray, *, lmbda: float) -> np.ndarray:
	"""Yeo-Johnson power transform at the power lmbda, which also works for values at or below 0."""
	return stats.yeojohnson(x.ravel(), lmbda=lmbda).reshape(-1, 1)

def _hinge(x: np.ndarray, *, t: float) -> np.ndarray:
	"""x and how far x is past t (0 below it), so the slope can change at t (see find_threshold.py)."""
	return np.hstack([x, np.maximum(0, x - t)])

def _cap(x: np.ndarray, *, t: float) -> np.ndarray:
	"""x up to t, then flat at t: the feature stops counting above t."""
	return np.minimum(x, t)

def _floor(x: np.ndarray, *, t: float) -> np.ndarray:
	"""Flat at t up to t, then x: the feature only counts above t."""
	return np.maximum(x, t)

def _winsorize(x: np.ndarray, *, low: float, high: float) -> np.ndarray:
	"""x clipped to [low, high]: the most extreme values on each side are pulled in to the clip points."""
	return np.clip(x, low, high)

def _step(x: np.ndarray, *, t: float) -> np.ndarray:
	"""0 below t, 1 at or above it: one level on each side of t."""
	return (x >= t).astype(float)


def _line(*steps) -> Pipeline:
	"""The given steps followed by a linear regression."""
	return make_pipeline(*steps, LinearRegression())

def _transform(func, **settings) -> FunctionTransformer:
	"""Wrap a transform above, with its settings, as a pipeline step."""
	return FunctionTransformer(func, kw_args=settings)

def _thresholds(x: np.ndarray) -> np.ndarray:
	"""Thresholds tried by the hinge, cap, floor and step: every distinct value that leaves at least MIN_GROUP_FRAC of
	the rows on each side, the same candidates find_threshold.py tries (see its comments for why)."""
	min_rows = math.ceil(MIN_GROUP_FRAC * len(x))
	values = np.unique(x)
	n_low = np.searchsorted(np.sort(x), values, side='left')  # rows below each value
	return values[(n_low >= min_rows) & (len(x) - n_low >= min_rows)]

def _power_label(shift: float, *, reflect: bool) -> str:
	"""What the power is applied to, ex 'x + 0.5', or 'reflected: 11.7 - x' when reflected."""
	if reflect:
		return f'reflected: {shift:.3g} - x' if shift else 'reflected: -x'
	return f'x {"+" if shift > 0 else "-"} {abs(shift):.3g}' if shift else 'x'

def _candidates(x: np.ndarray) -> dict[str, tuple[Family, list[tuple[str, Pipeline]]]]:
	"""Every candidate shape, keyed by name, before fitting.

	Each value is the candidate's family and its options: one (setting, unfitted pipeline) pair per value of its setting.
	"""
	span = x.max() - x.min()
	thresholds = _thresholds(x)

	def power(p: float, *, zero_ok: bool) -> list[tuple[str, Pipeline]]:
		# Both directions are tried. As it is (x + shift), the curve bends hardest near the smallest values. Reflected
		# (shift - x, the values counted down from the top), it bends hardest near the largest values, the mirror image.
		# For each direction, the shifts put the value at the bending end from 0.1% of the range up to 10 times the
		# range above 0 (exactly 0 too when the transform allows it, as sqrt does). The closer to 0, the harder the
		# bend, and the further away, the closer the curve gets to a straight line. As it is, no shift at all is also
		# tried when every value is above 0, which is the plain transform of x.
		options = []
		for reflect in (False, True):
			base = -x if reflect else x
			shifts = [step * span - base.min() for step in ((0,) if zero_ok else ()) + SHIFT_STEPS]
			if not reflect and x.min() > 0:
				shifts.insert(0, 0.0)
			options += [
				(_power_label(shift, reflect=reflect), _line(_transform(_power, power=p, shift=shift, reflect=reflect)))
				for shift in shifts
			]
		return options

	return {
		# re-expression
		'reciprocal': ('re-expression', power(-1, zero_ok=False)),
		'log': ('re-expression', power(0, zero_ok=False)),
		'sqrt': ('re-expression', power(0.5, zero_ok=True)),
		'Yeo-Johnson': ('re-expression', [
			(f'λ = {lmbda:g}', _line(_transform(_yeo_johnson, lmbda=lmbda))) for lmbda in YEO_JOHNSON_LAMBDAS
		]),

		# piecewise linear
		'hinge': ('piecewise linear', [(f't = {t:.3g}', _line(_transform(_hinge, t=t))) for t in thresholds]),
		'cap': ('piecewise linear', [(f't = {t:.3g}', _line(_transform(_cap, t=t))) for t in thresholds]),
		'floor': ('piecewise linear', [(f't = {t:.3g}', _line(_transform(_floor, t=t))) for t in thresholds]),
		'winsorize': ('piecewise linear', [
			(f'{p}% each side', _line(_transform(_winsorize, low=np.percentile(x, p), high=np.percentile(x, 100 - p))))
			for p in WINSORIZE_PERCENTS
		]),

		# bins
		'step': ('bins', [(f't = {t:.3g}', _line(_transform(_step, t=t))) for t in thresholds]),
		'quantile bins': ('bins', [(f'{N_BINS} bins', _line(KBinsDiscretizer(N_BINS, encode='onehot-dense', strategy='quantile', subsample=None)))]),
	}

def _columns(model: Pipeline, X: np.ndarray) -> int:
	"""Number of input columns a fitted pipeline uses, not counting the intercept."""
	# Count only the columns that add something: the rank of the columns next to the intercept's column of 1s, minus
	# that column. Ex 5 one-hot bins always add up to 1, so one of them repeats the intercept and they count as 4.
	columns = model[:-1].transform(X)
	return int(np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), columns]))) - 1

def _xy(*, df: pd.DataFrame, feature: str, target: str) -> tuple[np.ndarray, np.ndarray]:
	"""Pull the feature (x) and target (y) out of the DataFrame as plain number arrays."""
	# a row missing either value cannot be fitted or graded, so it is dropped here once for everything below
	data = df[[feature, target]].dropna()
	return data[feature].to_numpy(dtype=float), data[target].to_numpy(dtype=float)


def lowess_fit(*, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
	"""Fit LOWESS of y on x.

	Returns:
		The LOWESS value at each row, in the rows' order, and its R².
	"""
	fitted = fit_lowess(y, x, return_sorted=False)
	return fitted, float(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))

def reference_r2(*, df: pd.DataFrame, feature: str, target: str) -> dict[str, float]:
	"""R² of the straight line, the floor a candidate has to beat, and of LOWESS, the shape it is trying to reach.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Column the fits use.
		target: Column being predicted.

	Returns:
		R² keyed by 'linear' and 'LOWESS'.
	"""
	x, y = _xy(df=df, feature=feature, target=target)
	return {'linear': float(stats.linregress(x, y).rvalue ** 2), 'LOWESS': lowess_fit(x=x, y=y)[1]}

def shape_fits(*, df: pd.DataFrame, feature: str, target: str) -> list[ShapeFit]:
	"""Fit every candidate shape of a feature to the target, keeping the best searched setting of each.

	Fit it on training data only: the searched settings are picked on the same rows they are scored on.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Column the shapes are built from.
		target: Column being predicted.

	Returns:
		One ShapeFit per candidate, in the order of the menu above.
	"""
	x, y = _xy(df=df, feature=feature, target=target)
	X = x.reshape(-1, 1)  # sklearn expects a table with one column per input, so the feature becomes one column

	fits = []
	for name, (family, options) in _candidates(x).items():
		# fit every option and keep the one with the highest R²
		best = None
		for setting, model in options:
			r_squared = model.fit(X, y).score(X, y)
			# best[1] is the highest R² seen so far
			if best is None or r_squared > best[1]:
				best = (setting, r_squared, model)

		setting, r_squared, model = best
		fits.append(ShapeFit(name=name, family=family, setting=setting, columns=_columns(model, X), r_squared=float(r_squared), model=model))

	return fits

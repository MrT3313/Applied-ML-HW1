"""Fit many candidate shapes of one feature to the target, to find the ones that beat a straight line.

The big picture:
	A straight line assumes the target moves at the same rate across the whole range of a feature. The LOWESS curve
	on the scatter plot follows the local average of the target without assuming any shape, so it hints at the shape
	the relationship really has. Every candidate here is one standard way of rebuilding the feature as the input
	columns of an ordinary linear regression, so the fitted line can follow a bend. A candidate is worth carrying
	forward when it beats the straight line's R² and gets close to the LOWESS curve.

The three families of a continuous feature:
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
	shift and direction of a log. Every value in that setting's grid is tried and the one with the highest R² is kept.
	The search sees the same rows it is scored on, so the kept R² is a little optimistic.
	The number of bins is fixed instead: more bins means more columns, which almost always raises R² on these rows,
	so a search would always pick the largest number tried.

Notes for reading the results:
	- reciprocal, log and sqrt are the Box-Cox transform at the powers -1, 0 and 0.5. A straight line fits
	  (x^p - 1) / p exactly as well as x^p (one is a rescaled copy of the other), so Box-Cox at those powers is not
	  repeated as its own candidate.
	- reciprocal, log and sqrt each try both directions: as it is (steep at the low end, flat at the high end) and
	  reflected (the mirror image). Yeo-Johnson needs no reflected version: flipping the sign of the feature gives
	  exactly the fit of Yeo-Johnson at the power 2 - λ, so its powers above 1 already cover the mirror image.

Categorical features (category_fits):
	The same screen for a categorical feature stored as integer codes, ex Season 1 to 4. The baseline is the code
	used as a number on a straight line. The ceiling is one-hot, every level its own group, which predicts each row
	with the mean of its level (its R² is also called eta², the level means R²). A candidate is worth carrying forward
	when it beats the straight line and gets close to one-hot with fewer columns.

	The menu:
	- groups: one-hot, and the levels merged into 2 or 3 groups. The levels are sorted by their mean target and every
	  way to cut that sorted list into contiguous groups is tried.
	- piecewise linear: hinge, cap and floor on the code, at every level except the lowest and the highest.
	- cyclic (only for a feature that wraps around, ex Month 12 is next to Month 1): a sine and a cosine of the code
	  with 1 wave per period, or 2 waves (4 columns).
	A candidate that uses as many columns as one-hot (one per level, minus 1) is dropped: it fits exactly the same
	values as one-hot, ex a hinge on 3 levels.

	Graded by adjusted R² instead of R²: the candidates spend from 1 to 11 columns, and every column raises R² a
	little even when it is pure noise. On 584 rows, pure noise is worth about columns / (rows - 1) of R², ex about
	0.02 for the 11 columns of a Month one-hot, a full tier of gain. The continuous menu spends at most 4 columns,
	so plain R² is fair there. Adjusted R² charges every column that price, so a column of pure noise earns 0 on
	average.

	A control (ex Year) can be added to every fit, the baseline and one-hot included, so every candidate is graded
	on what it adds on top of the control. The groups are still searched on the plain level means, which ignore the
	control: the cut is exactly the best one without a control and close to it, but not always exactly it, with one.

	Left out on purpose:
	- step: every step on the levels is one of the 2 group cuts, and the best 2 way split always cuts the levels
	  sorted by their mean, so the 2 groups candidate already covers it.
	- re-expressions (log, sqrt, ...): bending a level code has no meaning, the codes are labels in an order.
	- sum, Helmert, difference and polynomial contrasts: the same R² as one-hot, only the coefficients read differently.
	- target and frequency encoding: an in-sample score leaks the target (each level's mean is already in the column),
	  hides k fitted means behind 1 column, and the frequency counts only reflect where the training dates stop.
	- ridge: a way of fitting, not an encoding of the feature.
	- periodic splines: left out of this first pass, as splines are left out of the continuous screen.
	- cross validation: the same in-sample approach as the continuous screen.

	The searched settings (the group cuts and the thresholds) make every R² a little optimistic: adjusted R² charges
	the columns, not the number of settings tried.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Literal, Optional, get_args

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from statsmodels.nonparametric.smoothers_lowess import lowess as fit_lowess

Family = Literal['re-expression', 'piecewise linear', 'bins', 'groups', 'cyclic']
FAMILIES: tuple[Family, ...] = get_args(Family)

# Searched settings (each value is tried, the best R² is kept)
MIN_GROUP_FRAC = 0.05  # hinge, cap, floor and step thresholds leave at least 5% of rows on each side
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
		columns: Number of input columns the fit uses, not counting the intercept or the control.
		rows: Number of rows it was fitted to.
		control_columns: Number of columns of the control added to every fit (see category_fits), 0 without one.
		r_squared: R² on the rows it was fitted to. Optimistic when a setting was searched.
		model: The fitted pipeline, which takes the feature (and the control, when there is one) and predicts the target.
	"""
	name: str
	family: Family
	setting: str
	columns: int
	rows: int
	control_columns: int = 0
	r_squared: float
	model: Pipeline

	@property
	def label(self) -> str:
		"""The name, followed by the setting in parentheses when the shape has one."""
		return f'{self.name} ({self.setting})' if self.setting else self.name

	@property
	def adjusted_r_squared(self) -> float:
		"""R² minus a charge for every column, the control's included (see adjusted_r2)."""
		return adjusted_r2(r_squared=self.r_squared, rows=self.rows, columns=self.columns + self.control_columns)

	def predict(self, *, x: np.ndarray, control: Optional[np.ndarray] = None) -> np.ndarray:
		"""Evaluate the fit at x, with the control at the given values when the fit has one."""
		X = np.asarray(x, dtype=float).reshape(-1, 1)
		if control is not None:
			X = np.column_stack([X, np.asarray(control, dtype=float)])  # the control is the second column, as in the fit
		return self.model.predict(X)


def adjusted_r2(*, r_squared: float, rows: int, columns: int) -> float:
	"""R² minus a charge for every column the fit uses.

	Every column raises R² a little, even a column of pure noise. Adjusted R² takes that expected rise back out, so a
	column of pure noise earns 0 on average and only a column that explains more than noise raises it.

	Args:
		r_squared: R² on the rows the fit was fitted to.
		rows: Number of rows it was fitted to.
		columns: Number of input columns it uses, not counting the intercept.

	Returns:
		The adjusted R².
	"""
	return 1 - (1 - r_squared) * (rows - 1) / (rows - columns - 1)


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
	"""x and how far x is past t (0 below it), so the slope can change at t."""
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

def _groups(x: np.ndarray, *, groups: list[list[float]]) -> np.ndarray:
	"""One 0/1 column per group of levels but the first: 1 on the rows whose level is in that group. One-hot is every
	level alone."""
	# The first group is the baseline: its rows are 0 in every column and the intercept fits their mean. With a column
	# for every group, the columns would add up to the intercept's column of 1s, and LinearRegression then fits a little
	# worse than it should (sklearn 1.3 gives a lower, even negative, R² for a Weekday one-hot built that way).
	return np.column_stack([np.isin(x.ravel(), group) for group in groups[1:]]).astype(float)

def _drop_first(columns: np.ndarray) -> np.ndarray:
	"""Every column but the first, so 0/1 bin columns no longer add up to the intercept's column of 1s (see _groups)."""
	return columns[:, 1:]

def _sin_cos(x: np.ndarray, *, period: float, waves: int) -> np.ndarray:
	"""A sine and a cosine of x for each of 1 to `waves` waves per period, so the last level wraps around to the first."""
	angles = [2 * np.pi * wave * x / period for wave in range(1, waves + 1)]
	return np.hstack([f(angle) for angle in angles for f in (np.sin, np.cos)])


def _line(*steps) -> Pipeline:
	"""The given steps followed by a linear regression."""
	return make_pipeline(*steps, LinearRegression())

def _transform(func, **settings) -> FunctionTransformer:
	"""Wrap a transform above, with its settings, as a pipeline step."""
	return FunctionTransformer(func, kw_args=settings)

def _thresholds(x: np.ndarray) -> np.ndarray:
	"""Thresholds tried by the hinge, cap, floor and step: every distinct value that leaves at least MIN_GROUP_FRAC of
	the rows on each side, so a threshold near the edge cannot win by isolating a few outliers."""
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
		'quantile bins': ('bins', [(f'{N_BINS} bins', _line(
			KBinsDiscretizer(N_BINS, encode='onehot-dense', strategy='quantile', subsample=None),
			_transform(_drop_first),
		))]),
	}

def _category_line(step) -> Pipeline:
	"""The step applied to the feature (the first column) while the control columns after it pass through unchanged,
	followed by a linear regression. Without a control, the feature is the only column."""
	return make_pipeline(ColumnTransformer([('shape', step, [0])], remainder='passthrough'), LinearRegression())

def _group_label(groups: list[list[float]]) -> str:
	"""The groups of levels, ex '{1, 4} | {2, 3}'."""
	return ' | '.join('{' + ', '.join(f'{level:g}' for level in sorted(group)) + '}' for group in groups)

def _category_candidates(
	x: np.ndarray,
	y: np.ndarray,
	*,
	period: Optional[float],
) -> dict[str, tuple[Family, list[tuple[str, Pipeline]]]]:
	"""Every candidate encoding of a categorical feature, keyed by name, before fitting (the same shape as _candidates)."""
	levels = np.unique(x)
	# the levels from the lowest mean target to the highest, the order the groups are cut in
	by_mean = list(pd.Series(y).groupby(x).mean().sort_values().index)
	thresholds = levels[1:-1]  # every level but the lowest and the highest, so each side keeps at least one level

	def groups(n: int) -> list[tuple[str, Pipeline]]:
		# every way to cut the levels sorted by mean into n contiguous groups: choose the n - 1 cut points
		options = []
		for cuts in itertools.combinations(range(1, len(by_mean)), n - 1):
			bounds = [0, *cuts, len(by_mean)]
			cut = [by_mean[start:end] for start, end in zip(bounds, bounds[1:])]
			options.append((_group_label(cut), _category_line(_transform(_groups, groups=cut))))
		return options

	menu = {
		# groups
		'one-hot': ('groups', [('', _category_line(_transform(_groups, groups=[[level] for level in levels])))]),
		'2 groups': ('groups', groups(2)),
		'3 groups': ('groups', groups(3)),

		# piecewise linear
		'hinge': ('piecewise linear', [(f't = {t:g}', _category_line(_transform(_hinge, t=t))) for t in thresholds]),
		'cap': ('piecewise linear', [(f't = {t:g}', _category_line(_transform(_cap, t=t))) for t in thresholds]),
		'floor': ('piecewise linear', [(f't = {t:g}', _category_line(_transform(_floor, t=t))) for t in thresholds]),
	}
	if period:
		for waves in (1, 2):
			menu[f'sine / cosine, {waves} wave{"" if waves == 1 else "s"}'] = (
				'cyclic', [('', _category_line(_transform(_sin_cos, period=period, waves=waves)))]
			)

	return {name: candidate for name, candidate in menu.items() if candidate[1]}  # drop a candidate with no option, ex 3 groups of 2 levels

def _rank(*blocks: np.ndarray) -> int:
	"""Number of columns of the blocks put side by side that are not a mix of the others."""
	return int(np.linalg.matrix_rank(np.column_stack(blocks)))

def _columns(model: Pipeline, X: np.ndarray) -> int:
	"""Number of input columns a fitted pipeline uses, not counting the intercept or the control."""
	# Count only the columns that add something: the rank of the columns next to the intercept's column of 1s, minus
	# that column. Ex 5 one-hot bins always add up to 1, so one of them repeats the intercept and they count as 4.
	# The control (every column of X after the feature, none for a continuous feature) is added to every fit, so its
	# columns are taken out the same way.
	ones = np.ones(len(X))
	return _rank(ones, model[:-1].transform(X)) - _rank(ones, X[:, 1:])

def _xy(*, df: pd.DataFrame, feature: str, target: str, control: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
	"""Pull the inputs (X) and target (y) out of the DataFrame as plain number arrays.

	X is a table with the feature as its first column, followed by the control when there is one.
	"""
	# a row missing any value cannot be fitted or graded, so it is dropped here once for everything below
	inputs = [feature] + ([control] if control else [])
	data = df[inputs + [target]].dropna()
	return data[inputs].to_numpy(dtype=float), data[target].to_numpy(dtype=float)

def _fit_best(
	candidates: dict[str, tuple[Family, list[tuple[str, Pipeline]]]],
	X: np.ndarray,
	y: np.ndarray,
) -> list[ShapeFit]:
	"""Fit every option of every candidate and keep the option with the highest R² of each."""
	control_columns = _rank(np.ones(len(X)), X[:, 1:]) - 1  # 0 without a control

	fits = []
	for name, (family, options) in candidates.items():
		best = None
		for setting, model in options:
			r_squared = model.fit(X, y).score(X, y)
			# best[1] is the highest R² seen so far
			if best is None or r_squared > best[1]:
				best = (setting, r_squared, model)

		setting, r_squared, model = best
		fits.append(ShapeFit(
			name=name,
			family=family,
			setting=setting,
			columns=_columns(model, X),
			rows=len(y),
			control_columns=control_columns,
			r_squared=float(r_squared),
			model=model,
		))

	return fits


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
	X, y = _xy(df=df, feature=feature, target=target)
	x = X[:, 0]
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
	X, y = _xy(df=df, feature=feature, target=target)  # X is a table with the feature as its only column
	return _fit_best(_candidates(X[:, 0]), X, y)

def category_reference_r2(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	control: Optional[str] = None,
) -> dict[str, float]:
	"""Adjusted R² of the straight line on the level code, the floor a candidate has to beat, and of one-hot, the
	ceiling it is trying to reach.

	Args:
		df: DataFrame holding the columns. Rows missing any of them are dropped.
		feature: Categorical column stored as numbers, ex Season 1 to 4.
		target: Column being predicted.
		control: Column added to both fits as it is, ex Year. When omitted, there is no control.

	Returns:
		Adjusted R² keyed by 'linear' and 'one-hot'.
	"""
	X, y = _xy(df=df, feature=feature, target=target, control=control)
	candidates = {
		'linear': ('piecewise linear', [('', _category_line('passthrough'))]),  # the code itself as one column (the family is never read)
		'one-hot': _category_candidates(X[:, 0], y, period=None)['one-hot'],
	}
	return {fit.name: fit.adjusted_r_squared for fit in _fit_best(candidates, X, y)}

def category_fits(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	period: Optional[float] = None,
	control: Optional[str] = None,
) -> list[ShapeFit]:
	"""Fit every candidate encoding of a categorical feature to the target, keeping the best searched setting of each.

	Fit it on training data only: the searched settings are picked on the same rows they are scored on. Grade the
	results by adjusted_r_squared (see the module docstring).

	Args:
		df: DataFrame holding the columns. Rows missing any of them are dropped.
		feature: Categorical column stored as numbers, ex Season 1 to 4.
		target: Column being predicted.
		period: Number of levels in one full cycle, ex 12 for Month, for a feature that wraps around. The cyclic
			candidates are left out when omitted.
		control: Column added to every fit as it is, ex Year, so each candidate is graded on what it adds on top of
			it. Predict with the control as well (ShapeFit.predict). When omitted, there is no control.

	Returns:
		One ShapeFit per candidate, in the order of the menu, leaving out every candidate that uses as many columns as
		one-hot (it fits exactly the same values).
	"""
	X, y = _xy(df=df, feature=feature, target=target, control=control)
	fits = _fit_best(_category_candidates(X[:, 0], y, period=period), X, y)
	one_hot = fits[0].columns  # one-hot is first in the menu
	return [fit for fit in fits if fit.name == 'one-hot' or fit.columns != one_hot]

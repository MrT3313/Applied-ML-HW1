import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from Q1.consts import SEEDS
from Q1.utils.scoring.scorers import SCORERS

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.02), 2)
MIN_GROUP_FRAC = 0.05


class GuardedGridSearch(BaseEstimator, RegressorMixin):
	"""Grid search over one parameter (or a tuple of parameters searched together) whose grid is filtered on each
	training fold before the search.

	Args:
		pipeline: Pipeline holding the parameter.
		param: Pipeline parameter to search, or a tuple of parameters searched together.
		grid: Candidate values. When param is a tuple, each item is a tuple of the same length, one value per parameter.
		random_state: random_state of the shuffled inner KFold.
		n_splits: Number of inner folds.
		min_group_frac: Smallest share of the training rows allowed on each side of a threshold t (x < t and
			x >= t). When param is a tuple, every value in a grid item must pass. Needs a single column X. None
			searches the whole grid, for parameters that are not thresholds.
	"""

	def __init__(
		self,
		pipeline: Pipeline,
		*,
		param: str | tuple[str, ...],
		grid: np.ndarray | list[float] | list[tuple[float, ...]],
		random_state: int,
		n_splits: int,
		min_group_frac: float | None,
	):
		self.pipeline = pipeline
		self.param = param
		self.grid = grid
		self.random_state = random_state
		self.n_splits = n_splits
		self.min_group_frac = min_group_frac

	def fit(self, X, y):
		grid = list(self.grid)
		if self.min_group_frac is not None:
			x = np.asarray(X, dtype=float)
			assert x.ndim == 2 and x.shape[1] == 1, "the group guard needs a single column X"
			x = x.ravel()

			def passes(t: float) -> bool:
				return min(np.mean(x < t), np.mean(x >= t)) >= self.min_group_frac

			if isinstance(self.param, tuple):
				grid = [values for values in grid if all(passes(t) for t in values)]
			else:
				grid = [t for t in grid if passes(t)]
			assert grid, "no threshold in the grid leaves min_group_frac of the rows on each side"

		if isinstance(self.param, tuple):
			param_grid = [{p: [v] for p, v in zip(self.param, values)} for values in grid]
		else:
			param_grid = {self.param: grid}

		self.grid_ = grid
		self.search_ = GridSearchCV(
			self.pipeline,
			param_grid=param_grid,
			scoring=SCORERS["rmse"],
			cv=KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state),
			n_jobs=1,  # the outer cross validation already runs folds in parallel
			refit=True,
		).fit(X, y)
		self.best_params_ = self.search_.best_params_
		return self

	def predict(self, X):
		return self.search_.predict(X)


def nested_threshold(
	pipeline: Pipeline,
	*,
	param: str | tuple[str, ...],
	grid: np.ndarray | list[float] | list[tuple[float, ...]] = THRESHOLD_GRID,
	random_state: int = SEEDS[0],
	n_splits: int = 4,
	min_group_frac: float | None = MIN_GROUP_FRAC,
) -> GuardedGridSearch:
	"""Wrap a pipeline in a grid search over one parameter (ex a threshold), or a tuple of parameters searched
	together, to be run inside each outer training fold.

	Args:
		pipeline: Pipeline holding the parameter.
		param: Pipeline parameter to search, ex "features__t" or "features__n_knots", or a tuple of parameters
			searched together, ex ("features__lo", "features__hi").
		grid: Candidate values. When param is a tuple, each item is a tuple of the same length, one value per
			parameter, ex (lo, hi) pairs.
		random_state: random_state of the shuffled inner KFold.
		n_splits: Number of inner folds.
		min_group_frac: Smallest share of the training fold's rows allowed on each side of a threshold, the same
			guard as Q1B's find_threshold. When param is a tuple, every value in a grid item must pass. None turns
			it off, for parameters that are not thresholds.

	Returns:
		An unfitted search that picks the value with the lowest inner RMSE and refits on the whole training fold.
	"""
	return GuardedGridSearch(
		pipeline, param=param, grid=grid, random_state=random_state, n_splits=n_splits, min_group_frac=min_group_frac,
	)


def picked_thresholds(
	estimators: list[GuardedGridSearch], *, param: str | tuple[str, ...],
) -> list[float] | list[tuple[float, ...]]:
	"""Read the value each fitted grid search picked for param, in fold order.

	Args:
		estimators: Fitted searches, one per outer fold.
		param: Searched parameter, or a tuple of parameters searched together.

	Returns:
		One float per fold, or one tuple of floats per fold (in the order of param) when param is a tuple.
	"""
	if isinstance(param, tuple):
		return [tuple(float(estimator.best_params_[p]) for p in param) for estimator in estimators]
	return [float(estimator.best_params_[param]) for estimator in estimators]

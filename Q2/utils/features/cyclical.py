import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class Cyclical(BaseEstimator, TransformerMixin):
	"""Replace each column x of codes on a cycle with [sin(2π·x / period), cos(2π·x / period)].

	The two outputs place every code on a circle, so the last code sits next to the first one (ex month 12
	next to month 1).

	Args:
		period: Number of codes in one full cycle, ex 12 for months.

	Attributes:
		n_features_in_: Number of input columns, set by fit.
	"""

	def __init__(self, period: float = 12):
		self.period = period

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "Cyclical":
		self.n_features_in_ = np.asarray(X).shape[1]
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		angle = 2 * np.pi * np.asarray(X, dtype=float) / self.period
		return np.hstack([np.sin(angle), np.cos(angle)])

	def get_feature_names_out(self, input_features: ArrayLike = None) -> np.ndarray:
		"""Name each output column, in the order transform returns them.

		Args:
			input_features: Input column names. None uses x0, x1, ... for the n_features_in_ columns seen by fit.

		Returns:
			"{col}__sin" for each input column, then "{col}__cos" for each input column.
		"""
		if input_features is None:
			input_features = [f"x{i}" for i in range(self.n_features_in_)]
		names = [f"{col}__sin" for col in input_features] + [f"{col}__cos" for col in input_features]
		return np.asarray(names, dtype=object)

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class Indicator(BaseEstimator, TransformerMixin):
	"""Replace each column x with the indicator 1.0 when x < t, else 0.0.

	Args:
		t: Split point. Rows with x < t are the low side, matching find_threshold.
		keep_raw: Output [x, indicator] instead of the indicator alone.

	Attributes:
		n_features_in_: Number of input columns, set by fit.
	"""

	def __init__(self, t: float = 0.5, keep_raw: bool = False):
		self.t = t
		self.keep_raw = keep_raw

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "Indicator":
		self.n_features_in_ = np.asarray(X).shape[1]
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		X = np.asarray(X, dtype=float)
		indicator = (X < self.t).astype(float)
		return np.hstack([X, indicator]) if self.keep_raw else indicator

	def get_feature_names_out(self, input_features: ArrayLike = None) -> np.ndarray:
		"""Name each output column, in the order transform returns them.

		Args:
			input_features: Input column names. None uses x0, x1, ... for the n_features_in_ columns seen by fit.

		Returns:
			"{col}__below_{t:g}" for each input column, after the raw column names when keep_raw.
		"""
		if input_features is None:
			input_features = [f"x{i}" for i in range(self.n_features_in_)]
		names = [f"{col}__below_{self.t:g}" for col in input_features]
		return np.asarray(list(input_features) + names if self.keep_raw else names, dtype=object)

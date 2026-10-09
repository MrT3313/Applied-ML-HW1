import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class Indicator(BaseEstimator, TransformerMixin):
	"""Replace each column x with the indicator 1.0 when x < t, else 0.0.

	Args:
		t: Split point. Rows with x < t are the low side, matching find_threshold.
		keep_raw: Output [x, indicator] instead of the indicator alone.
	"""

	def __init__(self, t: float = 0.5, keep_raw: bool = False):
		self.t = t
		self.keep_raw = keep_raw

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "Indicator":
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		X = np.asarray(X, dtype=float)
		indicator = (X < self.t).astype(float)
		return np.hstack([X, indicator]) if self.keep_raw else indicator

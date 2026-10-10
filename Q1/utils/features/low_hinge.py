import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class LowHinge(BaseEstimator, TransformerMixin):
	"""Replace each column x with the low-side hinge max(0, t - x), which is 0 at or above t and grows as x falls below it.

	Args:
		t: Hinge point.
		keep_raw: Output [x, max(0, t - x)] (a linear spline with a knot at t) instead of the hinge alone.
	"""

	def __init__(self, t: float = 0.5, keep_raw: bool = False):
		self.t = t
		self.keep_raw = keep_raw

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "LowHinge":
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		X = np.asarray(X, dtype=float)
		hinge = np.maximum(0.0, self.t - X)
		return np.hstack([X, hinge]) if self.keep_raw else hinge

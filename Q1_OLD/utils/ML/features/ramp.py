import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class Ramp(BaseEstimator, TransformerMixin):
	"""Replace each column x with the ramp clip(x, lo, hi), which is flat below lo, linear between lo and hi, and flat above hi.

	Args:
		lo: Lower corner. Values below it become lo.
		hi: Upper corner. Values above it become hi.
	"""

	def __init__(self, lo: float = 0.0, hi: float = 1.0):
		self.lo = lo
		self.hi = hi

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "Ramp":
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		return np.clip(np.asarray(X, dtype=float), self.lo, self.hi)

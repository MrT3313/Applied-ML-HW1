import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class Cyclical(BaseEstimator, TransformerMixin):
	"""Replace each column x of codes on a cycle with [sin(2π·x / period), cos(2π·x / period)].

	The two outputs place every code on a circle, so the last code sits next to the first one (ex month 12
	next to month 1).

	Args:
		period: Number of codes in one full cycle, ex 12 for months.
	"""

	def __init__(self, period: float = 12):
		self.period = period

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "Cyclical":
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		angle = 2 * np.pi * np.asarray(X, dtype=float) / self.period
		return np.hstack([np.sin(angle), np.cos(angle)])

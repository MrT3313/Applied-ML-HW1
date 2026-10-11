import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClip(BaseEstimator, TransformerMixin):
	"""Clip each column to its Tukey fences, Q1 - 1.5·IQR and Q3 + 1.5·IQR, learned from the training data.

	Attributes:
		lower_: Lower fence per column, set by fit.
		upper_: Upper fence per column, set by fit.
	"""

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "IQRClip":
		X = np.asarray(X, dtype=float)
		q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
		iqr = q3 - q1
		self.lower_ = q1 - 1.5 * iqr
		self.upper_ = q3 + 1.5 * iqr
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin


class MergeLevels(BaseEstimator, TransformerMixin):
	"""Replace levels of a categorical column with other levels, ex {3: 2} turns every 3 into a 2.

	Args:
		mapping: Level to replace, mapped to the level that takes its place. Levels not in the mapping are unchanged.
	"""

	def __init__(self, mapping: dict):
		self.mapping = mapping

	def fit(self, X: ArrayLike, y: ArrayLike = None) -> "MergeLevels":
		return self

	def transform(self, X: ArrayLike) -> np.ndarray:
		return pd.DataFrame(np.asarray(X)).replace(self.mapping).to_numpy()

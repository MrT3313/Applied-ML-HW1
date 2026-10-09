import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class MinShift(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
	# FunctionTransformer learns nothing in fit, so log / sqrt / fractional powers on a column with zeros or
	# negatives (ex: Generosity) return NaN. Put this in front of them: it learns each column's minimum on the
	# training fold and shifts the columns whose minimum is <= 0 up so that minimum becomes `floor`.
	# Columns that are already positive are left alone.
	# floor=1 before log (so log(min) = 0), floor=0 before sqrt and fractional powers (they accept 0)
	# A validation / test value below the training minimum would land below floor (and below 0 when floor=0,
	# which sqrt turns into NaN), so shifted columns are clipped at floor: anything under the training
	# minimum is treated as the training minimum

	def __init__(self, floor: float = 0.0):
		self.floor = floor

	def fit(self, X, y=None):
		# allow-nan so this also works before an imputer; nanmin skips the missing values
		X = self._validate_data(X, force_all_finite="allow-nan")
		col_min = np.nanmin(X, axis=0)
		self.shifted_ = col_min <= 0
		self.shift_ = np.where(self.shifted_, self.floor - col_min, 0.0)
		return self

	def transform(self, X):
		check_is_fitted(self)
		X = self._validate_data(X, force_all_finite="allow-nan", reset=False)
		X = X + self.shift_
		# np.maximum keeps NaN as NaN (np.fmax would replace it with floor)
		X[:, self.shifted_] = np.maximum(X[:, self.shifted_], self.floor)
		return X

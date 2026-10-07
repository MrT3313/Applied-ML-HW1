import inspect

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from typing import Any, Callable, Dict, Optional


class SeriesTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
	"""
		Use when:
			Running one of this package's series transforms inside a Pipeline, ColumnTransformer or CV search
		Does:
			Applies fn to every column separately. Transforms that take fit_series (box_cox, yeo_johnson,
			quantile, rank, clip, reflect_log, shift) learn from the columns seen in fit, so in cross-validation
			they only ever learn from the training fold
		Requires:
			fn takes the column as the keyword argument series and returns a Series
		Caution:
			Raises if fn turns a value into NaN or inf, so a domain error is never silently imputed later
			NaN in the input passes through untouched, so this can run before an imputer

		Example:
			SeriesTransformer(fn=power_transform, kw_args={"power": 0.25})
	"""

	def __init__(self, fn: Callable[..., Series], kw_args: Optional[Dict[str, Any]] = None):
		# sklearn's clone and set_params need __init__ to only store its arguments
		self.fn = fn
		self.kw_args = kw_args

	def fit(self, X, y=None):
		X = self._to_frame(X)
		if all(isinstance(c, str) for c in X.columns):
			self.feature_names_in_ = np.asarray(X.columns, dtype=object)
		self.n_features_in_ = X.shape[1]

		# Keep the training columns so learned transforms refit on them for every call to transform
		self.fit_df_ = X.copy()
		return self

	def transform(self, X):
		check_is_fitted(self, "fit_df_")
		X = self._to_frame(X)
		assert X.shape[1] == self.n_features_in_, f"fit saw {self.n_features_in_} columns, transform got {X.shape[1]}"

		kw_args = dict(self.kw_args or {})
		learns_from_train = "fit_series" in inspect.signature(self.fn).parameters

		out = X.copy()
		for col in X.columns:
			if learns_from_train:
				kw_args["fit_series"] = self.fit_df_[col]
			out[col] = self.fn(series=X[col], **kw_args)

			# numpy turns log(-1) into NaN with only a warning, which a later imputer would hide
			created_bad = X[col].notna() & ~np.isfinite(out[col])
			if created_bad.any():
				raise ValueError(f"{self.fn.__name__} produced NaN or inf for {created_bad.sum()} values in column {col!r}")

		return out

	def _to_frame(self, X) -> DataFrame:
		if isinstance(X, DataFrame):
			return X
		# Arrays get the names seen in fit, so fit_df_ columns line up in transform
		columns = getattr(self, "feature_names_in_", None)
		return DataFrame(X, columns=columns)

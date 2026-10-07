import numpy as np
from pandas import Series

from typing import Optional


def sqrt_transform(*, series: Series, fit_series: Optional[Series] = None) -> Series:
	"""
		Use when:
			Count data or log is too strong
		Fixes:
			Mild right skew
		Requires:
			Any x (negatives shift up so the train min is 0)
		Caution:
			The shift is learned, so pass the TRAIN data as fit_series when transforming test data
	"""
	fit_series = series if fit_series is None else fit_series

	# Only shift when the train data has negatives
	series = series + max(0, -fit_series.min())

	assert series.min() >= 0, "sqrt requires x >= 0 (test value below the train min)"
	return np.sqrt(series)

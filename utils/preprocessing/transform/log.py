import numpy as np
from pandas import Series

from typing import Optional


def log_transform(*, series: Series, fit_series: Optional[Series] = None) -> Series:
	"""
		Use when:
			Values span orders of magnitude or when effects are multiplicative (10% more GDP matters not +$1000)
		Fixes:
			Strong right skew
		Requires:
			Any x (zeros and negatives shift up so the train min is 1)
		Caution:
			The shift is learned, so pass the TRAIN data as fit_series when transforming test data
	"""
	fit_series = series if fit_series is None else fit_series

	# Only shift when the train data has zeros or negatives, so log(min) = 0
	fit_min = fit_series.min()
	if fit_min <= 0:
		series = series + (1 - fit_min)

	# Series.min skips NaN and numpy keeps NaN, so missing values pass through untouched
	assert series.min() > 0, "log requires x > 0 (test value below the train min)"
	return np.log(series)

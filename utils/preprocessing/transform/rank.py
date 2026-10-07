import numpy as np
from pandas import Series

from typing import Optional


def rank_transform(*, series: Series, fit_series: Optional[Series] = None) -> Series:
	"""
		Use when:
			Only the order of values matters, or heavy outliers and odd shapes
			(this is the book's bucketizing: replace each value with its percentile)
		Fixes:
			Any skew or shape (output is the percentile rank between 0 and 1)
		Requires:
			Any x
		Caution:
			Throws away all magnitude information
			Ranks are learned, so pass the TRAIN data as fit_series when transforming test data
	"""
	fit_series = series if fit_series is None else fit_series
	sorted_fit = np.sort(fit_series.dropna().to_numpy())

	# Fraction of training values <= each value, so test values are ranked against train
	ranks = np.searchsorted(sorted_fit, series.to_numpy(), side="right") / len(sorted_fit)

	return Series(ranks, index=series.index, name=series.name).where(series.notna())

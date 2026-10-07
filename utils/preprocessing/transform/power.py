from pandas import Series

from typing import Optional


def power_transform(*, series: Series, power: float, fit_series: Optional[Series] = None) -> Series:
	"""
		Use when:
			Heavy right tail (0 < power < 1, smaller is stronger, sqrt is 0.5)
		Fixes:
			Right skew when 0 < power < 1, left skew when power > 1
		Requires:
			Any x (fractional powers shift negatives up so the train min is 0)
		Caution:
			The shift is learned, so pass the TRAIN data as fit_series when transforming test data
	"""
	fit_series = series if fit_series is None else fit_series

	# Only fractional powers need x >= 0, so only they shift
	if not float(power).is_integer():
		series = series + max(0, -fit_series.min())
		assert series.min() >= 0, "fractional powers require x >= 0 (test value below the train min)"

	return series ** power

from pandas import Series
from sklearn.preprocessing import PowerTransformer

from typing import Optional


def yeo_johnson_transform(*, series: Series, fit_series: Optional[Series] = None) -> Series:
	"""
		Use when:
			The safe default for any numeric column, including zeros and negatives
		Fixes:
			Left and right skew (learns a power lambda: 1 = none, 0 = log-like, > 1 = fixes left skew)
		Requires:
			Any x
		Caution:
			Lambda is learned, so pass the TRAIN data as fit_series when transforming test data
	"""
	fit_series = series if fit_series is None else fit_series

	# standardize=False keeps scaling as its own step. sklearn needs 2D input and ignores NaN when fitting
	transformer = PowerTransformer(method="yeo-johnson", standardize=False)
	transformer.fit(fit_series.to_numpy().reshape(-1, 1))
	transformed = transformer.transform(series.to_numpy().reshape(-1, 1)).ravel()

	return Series(transformed, index=series.index, name=series.name)

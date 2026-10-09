from pandas import Series
from sklearn.preprocessing import KBinsDiscretizer

from typing import Literal, Optional
from Q1_OLD_OLD.consts import HIST_BINS

def bucket_transform(
	*,
	series: Series,
	n_buckets: int = HIST_BINS,
	strategy: Literal["quantile", "uniform", "kmeans"] = "quantile",
	fit_series: Optional[Series] = None,
) -> Series:
	"""
		Use when:
			Heavy tail, or the relationship with the target is curved or stepped
		Fixes:
			Skew (quantile gives a roughly uniform result), and non-linear shapes when the buckets are one-hot encoded
		Requires:
			Any x
		Strategies:
			quantile: same number of rows per bucket (4 buckets = quartiles)
			uniform:  same width per bucket, evenly spaced between min and max
			kmeans:   edges at the natural gaps between clusters
		Caution:
			Output is the bucket number (0 to n_buckets - 1), so one-hot encode it to treat buckets as categories
			Edges are learned, so pass the TRAIN data as fit_series when transforming test data
			Quantile drops duplicate edges (ex: many repeated values), which can leave fewer than n_buckets
	"""
	fit_series = series if fit_series is None else fit_series

	# encode="ordinal" returns the bucket number. sklearn needs 2D input and can't take NaN, so fit and transform without it
	discretizer = KBinsDiscretizer(n_bins=n_buckets, encode="ordinal", strategy=strategy, subsample=None)
	discretizer.fit(fit_series.dropna().to_numpy().reshape(-1, 1))

	# NaN passes through untouched. Test values outside the train range land in the first or last bucket
	buckets = series.copy()
	present = series.notna()
	buckets[present] = discretizer.transform(series[present].to_numpy().reshape(-1, 1)).ravel()

	return buckets

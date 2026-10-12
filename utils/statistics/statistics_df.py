from typing import Optional

import pandas as pd

from utils import rule_of_thumb


def statistics_df(*, series: pd.Series, target: Optional[pd.Series] = None, transpose: bool = False) -> pd.DataFrame:
	"""Compute statistics for a single column.

	Args:
		series: Column to summarize.
		target: Column the series is correlated against. Omit it when the series is the target itself.
		transpose: If True, return one column per statistic (horizontal) instead
			of one row per statistic (vertical).

	Returns:
		A one-column DataFrame named after the column, indexed by statistic:
		describe() output, missing counts, range, skew and excess kurtosis (with
		their rule-of-thumb labels), IQR fences, outlier counts, and (when `target` is given)
		correlation with the target. Transposed to a single row when `transpose` is True.
	"""
	stats = series.describe()

	skew = series.skew()
	kurt = series.kurt()

	q1, q3 = series.quantile(0.25), series.quantile(0.75)
	iqr = q3 - q1
	lower_fence = q1 - 1.5 * iqr
	upper_fence = q3 + 1.5 * iqr

	row = {
		'count': stats['count'],
		'missing': series.isna().sum(),
		'% missing': series.isna().mean() * 100,
		'mean': stats['mean'],
		'std': stats['std'],
		'min': stats['min'],
		'25%': stats['25%'],
		'50%': stats['50%'],
		'75%': stats['75%'],
		'max': stats['max'],
		'range': stats['max'] - stats['min'],
		'skew': skew,
		'abs(skew)': abs(skew),
		'skew RoT': rule_of_thumb.rule_of_thumb__skewness(value=skew),
		'excess kurtosis': kurt,
		'abs(excess kurtosis)': abs(kurt),
		'kurtosis RoT': rule_of_thumb.rule_of_thumb__kurtosis(value=kurt),
		'Q1': q1,
		'Q3': q3,
		'IQR': iqr,
		'IQR lower fence': lower_fence,
		'IQR upper fence': upper_fence,
		'outliers below': (series < lower_fence).sum(),
		'outliers above': (series > upper_fence).sum(),
	}
	if target is not None:
		row['corr with target'] = series.corr(target)

	df = pd.DataFrame([row], index=[series.name]).round(3)

	return df if transpose else df.T
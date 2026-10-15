from typing import Optional

import pandas as pd

from utils import RULE_OF_THUMB

# the statistics in the column's own units, rounded by the column's scale
IN_UNITS = ['mean', 'std', 'min', '50%', 'max', 'range', 'Q1', 'Q3', 'IQR', 'IQR lower fence', 'IQR upper fence']


def continuous_statistics_df(*, series: pd.Series, target: Optional[pd.Series] = None) -> pd.DataFrame:
	"""Compute statistics for a single column.

	Args:
		series: Column to summarize.
		target: Column the series is correlated against. Omit it when the series is the target itself.

	Returns:
		A one-column DataFrame named after the column, indexed by statistic: count and missing values, center, spread,
		quartiles, skew and excess kurtosis (with their rule-of-thumb labels), IQR fences, outlier counts, and (when
		`target` is given) correlation with the target. The statistics in the column's units are rounded to 3 decimals,
		or to 1 when the std is 100 or more (3 decimals of a count in the thousands is noise). The rest are rounded to 3.
	"""
	stats = series.describe()

	skew = series.skew()
	kurt = series.kurt()

	q1, q3 = series.quantile(0.25), series.quantile(0.75)
	iqr = q3 - q1
	lower_fence = q1 - 1.5 * iqr
	upper_fence = q3 + 1.5 * iqr

	row = {
		'count': series.count(),
		'missing': series.isna().sum(),
		'% missing': series.isna().mean() * 100,
		'mean': stats['mean'],
		'std': stats['std'],
		'min': stats['min'],
		'50%': stats['50%'],
		'max': stats['max'],
		'range': stats['max'] - stats['min'],
		'skew': skew,
		'skew RoT': RULE_OF_THUMB.rule_of_thumb__skewness(value=skew),
		'excess kurtosis': kurt,
		'kurtosis RoT': RULE_OF_THUMB.rule_of_thumb__kurtosis(value=kurt),
		'Q1': q1,
		'Q3': q3,
		'IQR': iqr,
		'IQR lower fence': lower_fence,
		'outliers below': (series < lower_fence).sum(),
		'IQR upper fence': upper_fence,
		'outliers above': (series > upper_fence).sum(),
	}
	if target is not None:
		row['corr with target'] = series.corr(target)

	unit_decimals = 1 if stats['std'] >= 100 else 3
	decimals = {name: unit_decimals if name in IN_UNITS else 3 for name in row}

	return pd.DataFrame([row], index=[series.name]).round(decimals).T
from typing import Optional

import pandas as pd


def categorical_statistics_df(*, df: pd.DataFrame, feature: str, target: str, by: Optional[str] = None) -> pd.DataFrame:
	"""Compute statistics of the target within each level of a categorical column.

	Args:
		df: DataFrame holding the columns. Rows missing any of them are dropped.
		feature: Categorical column that defines the levels.
		target: Column summarized within each level.
		by: Second categorical column that splits every level of feature. When omitted, the levels are not split.

	Returns:
		A DataFrame with one row per level, or per (feature level, by level) pair that has rows when `by` is given,
		sorted by level: describe() output of the target, the share of rows, and the mean minus the target mean
		over all rows.
	"""
	keys = [feature, by] if by else [feature]
	data = df[keys + [target]].dropna()

	out = data.groupby(keys)[target].describe()  # count, mean, std, min, 25%, 50%, 75%, max
	out['count'] = out['count'].astype(int)
	out.insert(1, '% of rows', out['count'] / len(data) * 100)
	out['mean - overall mean'] = out['mean'] - data[target].mean()

	return out.round(3)

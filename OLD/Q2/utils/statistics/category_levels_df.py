import pandas as pd


def level_means_r_squared(*, df: pd.DataFrame, feature: str, target: str) -> float:
	"""Share of the target's variance explained by predicting each row with the target mean of its level.

	This is the R² of a one-hot encoded linear fit of target on feature. It is scored on the same rows
	the level means come from, so it is optimistic.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Categorical column that defines the levels.
		target: Column being predicted.

	Returns:
		R² in [0, 1].
	"""
	data = df[[feature, target]].dropna()

	level_means = data.groupby(feature)[target].transform('mean')
	sse = ((data[target] - level_means) ** 2).sum()
	sst = ((data[target] - data[target].mean()) ** 2).sum()

	return float(1 - sse / sst)


def category_levels_df(*, df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
	"""Summarize the target within each level of a categorical column.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Categorical column that defines the levels.
		target: Column summarized within each level.

	Returns:
		A DataFrame with one row per level, sorted by level and rounded to 3 decimals:
			rows: rows at that level.
			% of rows: share of all rows used.
			{target} mean, median, std, min, max: the target within that level.
			mean - overall mean: the level's target mean minus the target mean over all rows used.
	"""
	data = df[[feature, target]].dropna()
	groups = data.groupby(feature)[target]

	out = pd.DataFrame({
		'rows': groups.size(),
		'% of rows': groups.size() / len(data) * 100,
		f'{target} mean': groups.mean(),
		f'{target} median': groups.median(),
		f'{target} std': groups.std(),
		f'{target} min': groups.min(),
		f'{target} max': groups.max(),
		'mean - overall mean': groups.mean() - data[target].mean(),
	}).round(3)

	return out.sort_index()

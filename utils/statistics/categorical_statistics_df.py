from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.statistics.shape_fits import adjusted_r2


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


def interaction_r2_df(*, df: pd.DataFrame, feature: str, target: str, continuous: list[str]) -> pd.DataFrame:
	"""Check, for each continuous attribute, whether its slope on the target changes from one level of a categorical
	column to the next.

	This is the one table that crosses attributes: every other screen looks at one attribute at a time. For each
	continuous attribute, two linear regressions are fitted and compared by adjusted R²:
		- without: the categorical column as one-hot (one level left out as the baseline) and the continuous attribute,
		  so every level shifts the target up or down but they all share one slope.
		- with: the same columns plus each one-hot column times the continuous attribute, so every level gets its own slope.
	The continuous attribute is used raw, without any of its candidate shapes.

	Args:
		df: DataFrame holding the columns. For each continuous attribute, rows missing it, the feature or the target
			are dropped.
		feature: Categorical column that defines the levels.
		target: Column being predicted.
		continuous: Continuous attributes to cross with the feature.

	Returns:
		A DataFrame with one row per continuous attribute: the adjusted R² without and with the slope per level, the
		gain (with minus without), and the number of columns the slope per level adds (one per level, minus 1).
	"""
	rows = {}
	for column in continuous:
		data = df[[feature, column, target]].dropna()
		one_hot = pd.get_dummies(data[feature], drop_first=True, dtype=float).to_numpy()  # one column per level but the baseline
		x = data[[column]].to_numpy(dtype=float)
		y = data[target].to_numpy(dtype=float)

		adjusted = {}
		for name, X in (('without', np.hstack([one_hot, x])), ('with', np.hstack([one_hot, x, one_hot * x]))):
			r_squared = LinearRegression().fit(X, y).score(X, y)
			adjusted[name] = adjusted_r2(r_squared=r_squared, rows=len(y), columns=X.shape[1])

		rows[column] = {
			'adjusted r2 without': adjusted['without'],
			'adjusted r2 with': adjusted['with'],
			'gain': adjusted['with'] - adjusted['without'],
			'columns added': one_hot.shape[1],
		}

	out = pd.DataFrame.from_dict(rows, orient='index')
	out.index.name = 'continuous attribute'

	return out.round(4)

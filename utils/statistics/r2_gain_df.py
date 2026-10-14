from itertools import combinations

import pandas as pd
from sklearn.linear_model import LinearRegression


def _r2(X: pd.DataFrame, y: pd.Series) -> float:
	"""Fit a linear regression of y on X and return its R² on the same rows."""
	return LinearRegression().fit(X, y).score(X, y)


def r2_gain_df(*, df: pd.DataFrame, target: str, size: int = 2) -> pd.DataFrame:
	"""Compute how much R² with the target each combination of `size` attributes gains over its best smaller subset.

	Each combination is scored by the R² of a linear regression of the target on its attributes. The baseline is
	the best R² among its subsets with one attribute left out: the best single attribute for size 2, the best pair
	for size 3. The gain is what the weakest attribute of the combination adds on top of the others.

	The regressions are fit and scored on the same rows, so R² can never go down when an attribute is added. Every
	gain is 0 or higher and optimistic.

	Args:
		df: DataFrame holding the attributes and the target. Each combination uses the rows where all of its
			attributes and the target are present.
		target: Column the attributes are meant to predict.
		size: Number of attributes in each combination. Must be at least 2.

	Returns:
		A DataFrame with one row per combination, indexed by the attribute names joined with ' + ', sorted by
		r2 gain, largest first, and rounded to 3 decimals:
			r2 best with one left out: highest R² among the subsets with one attribute left out.
			r2 together: R² of all the combination's attributes.
			r2 gain: r2 together minus r2 best with one left out.
	"""
	rows = {}
	for attributes in combinations(df.columns.drop(target), size):
		complete = df[[*attributes, target]].dropna()
		y = complete[target]

		together = _r2(complete[list(attributes)], y)
		# every way to leave one attribute out, scored on the same rows as the full combination
		one_left_out = max(_r2(complete[list(subset)], y) for subset in combinations(attributes, size - 1))

		rows[' + '.join(attributes)] = {
			'r2 best with one left out': one_left_out,
			'r2 together': together,
			'r2 gain': together - one_left_out,
		}

	return pd.DataFrame.from_dict(rows, orient='index').sort_values('r2 gain', ascending=False).round(3)

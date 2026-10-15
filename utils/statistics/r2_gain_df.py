from itertools import combinations

import pandas as pd
from sklearn.linear_model import LinearRegression


def _r2(*, X: pd.DataFrame, y: pd.Series) -> float:
	"""Fit a linear regression of y on X and return its R² on the same rows."""
	return LinearRegression().fit(X, y).score(X, y)


def r2_gain_df(*, df: pd.DataFrame, target: str, min_gain: float) -> pd.DataFrame:
	"""Compute how much R² with the target each pair of attributes gains over the better of its two attributes alone.

	Each pair is scored by the R² of a linear regression of the target on both attributes. The baseline is the higher
	R² of the two single-attribute fits. The gain is what the weaker attribute of the pair adds on top of the other.

	The regressions are fit and scored on the same rows, so R² can never go down when an attribute is added. Every
	gain is 0 or higher and optimistic.

	Args:
		df: DataFrame holding the attributes and the target. Each pair uses the rows where both of its attributes and
			the target are present.
		target: Column the attributes are meant to predict.
		min_gain: Smallest r2 gain for a pair to be kept, compared after rounding. A line saying how many pairs were
			left out is printed.

	Returns:
		A DataFrame with one row per kept pair, indexed by the two attribute names joined with ' + ', sorted by r2 gain,
		largest first, and rounded to 3 decimals:
			r2 best with one left out: the higher R² of the two single-attribute fits.
			r2 together: R² of both attributes.
			r2 gain: r2 together minus r2 best with one left out.
	"""
	rows = {}
	for pair in combinations(df.columns.drop(target), 2):
		complete = df[[*pair, target]].dropna()
		y = complete[target]

		together = _r2(X=complete[list(pair)], y=y)
		# each attribute alone, scored on the same rows as the pair
		one_left_out = max(_r2(X=complete[[attribute]], y=y) for attribute in pair)

		rows[' + '.join(pair)] = {
			'r2 best with one left out': one_left_out,
			'r2 together': together,
			'r2 gain': together - one_left_out,
		}

	out = pd.DataFrame.from_dict(rows, orient='index').sort_values('r2 gain', ascending=False).round(3)
	kept = out[out['r2 gain'] >= min_gain]
	print(f'{len(kept)} of {len(out)} pairs shown: the other {len(out) - len(kept)} gain less than {min_gain:g}.')
	return kept

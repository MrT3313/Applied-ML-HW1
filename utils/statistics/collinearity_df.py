from itertools import combinations
from typing import Optional

import pandas as pd


def collinearity_df(*, df: pd.DataFrame, target: str, min_abs_r: Optional[float] = None) -> pd.DataFrame:
	"""Compute Pearson r for every pair of attributes, next to each attribute's Pearson r with the target.

	Collinearity is between the attributes only, so the target never forms a pair. Its r with each attribute is
	shown because it tells which attribute of a strongly correlated pair is the better one to keep.

	Args:
		df: DataFrame holding the attributes and the target. Each r uses the rows where both of its columns are
			present.
		target: Column the attributes are meant to predict.
		min_abs_r: Smallest size of the r between attributes for a pair to be kept, compared after rounding. A line
			saying how many pairs were left out is printed. None keeps every pair.

	Returns:
		A DataFrame with one row per pair of attributes, indexed by the two names joined with ' + ', sorted by the
		size of the r between the attributes, largest first, and rounded to 3 decimals:
			r between attributes: Pearson r of the two attributes with each other.
			r with target, first: Pearson r of the first named attribute with the target.
			r with target, second: Pearson r of the second named attribute with the target.
	"""
	corr = df.corr()
	attributes = corr.columns.drop(target)

	out = pd.DataFrame.from_dict(
		{
			f'{first} + {second}': {
				'r between attributes': corr.loc[first, second],
				'r with target, first': corr.loc[first, target],
				'r with target, second': corr.loc[second, target],
			}
			for first, second in combinations(attributes, 2)
		},
		orient='index',
	)

	out = out.sort_values('r between attributes', key=abs, ascending=False).round(3)
	if min_abs_r is None:
		return out

	kept = out[out['r between attributes'].abs() >= min_abs_r]
	print(f'{len(kept)} of {len(out)} pairs shown: the other {len(out) - len(kept)} have |r between attributes| below {min_abs_r:g}.')
	return kept

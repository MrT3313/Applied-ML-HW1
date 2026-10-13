from itertools import combinations

import numpy as np
import pandas as pd


def collinearity_df(*, df: pd.DataFrame, feature_pearson_threshold: float = 0.7, filter: bool = True) -> pd.DataFrame:
	"""Rank every pair of columns by how strongly they correlate with each other.

	A pair above the threshold says close to the same thing, so one of the two is redundant.
	This only compares columns with each other, so leave the target out of `df`.

	Args:
		df: Columns to compare.
		feature_pearson_threshold: |r| between two columns above which one is redundant.
			Dormann et al. (2013), Ecography 36: 27-46, found |r| > 0.7 is where collinearity
			starts to seriously distort a model.
		filter: If True, return only the redundant pairs.

	Returns:
		A DataFrame with one row per pair, indexed by the column names joined with ' + ':
			rows used: rows where both columns are present. The pair is scored on these rows.
			Pearson r between features: Pearson r of the two columns with each other.
			|r| between features > threshold: whether the pair is redundant.
			outcome: the suggested outcome.
		Sorted by |r|, highest first, and rounded to 3 decimals. The threshold is checked against
		the rounded r, so the flag matches what is shown.
	"""
	if df.shape[1] < 2:
		raise ValueError(f'df needs at least 2 columns, got {df.shape[1]}')

	rows = {}
	for feature_a, feature_b in combinations(df.columns, 2):
		complete = df[[feature_a, feature_b]].dropna()
		rows[f'{feature_a} + {feature_b}'] = {
			'rows used': len(complete),
			'Pearson r between features': complete[feature_a].corr(complete[feature_b]),
		}

	out = pd.DataFrame.from_dict(rows, orient='index').round(3)

	redundant = out['Pearson r between features'].abs() > feature_pearson_threshold
	out[f'|r| between features > {feature_pearson_threshold}'] = redundant
	out['outcome'] = np.where(redundant, 'redundant: keep one, or use regularization', 'not redundant')

	if filter:
		out = out[redundant]
	return out.sort_values('Pearson r between features', key=abs, ascending=False)

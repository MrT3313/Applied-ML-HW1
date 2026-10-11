from itertools import combinations

import numpy as np
import pandas as pd


def _fitted_correlations(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
	"""Merge X's columns into one, the target's least-squares fit on them, and correlate it with y.

	Returns:
		(Pearson r, Spearman rho) of the merged column with y. The Pearson r is the multiple
		correlation R; for a single column it equals that column's |r| with y.
	"""
	design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
	coefficients = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)[0]
	combined = pd.Series(design @ coefficients, index=y.index)
	return combined.corr(y), combined.corr(y, method='spearman')


def combination_gain_df(
	*,
	df: pd.DataFrame,
	target: pd.Series,
	size: int = 2,
	cohen_f2_threshold: float = 0.15,
	filter: bool = True,
) -> pd.DataFrame:
	"""Rank every combination of `size` columns by how much it adds over its best `size - 1` subset.

	Each combination is merged into one column: the fitted values of a linear regression of the
	target on its columns. That merged column is then correlated with the target. Its Pearson r
	is the multiple correlation R. Every `size - 1` subset of the combination is merged and scored
	the same way, and the one with the highest Pearson R is the baseline. For size 2 the baseline is the best
	single column; for size 3, the best pair; and so on. So the gain is what the last column adds
	on top of the best combination that leaves one column out.

	The gain is scored as Cohen's f² = (R² all features - R² baseline) / (1 - R² all features): the
	extra share of the target the combination explains, relative to what it still leaves unexplained.

	The regressions are fit and scored on the same rows, so the combined values are optimistic.
	Run it on training data only.

	Args:
		df: Columns to combine. A column named like the target is dropped.
		target: Column every combination is correlated against.
		size: How many columns go into each combination. Must be at least 2.
		cohen_f2_threshold: Smallest f² worth adding the extra column for.
			Cohen (1988): 0.02 small, 0.15 medium, 0.35 large.
		filter: If True, return only the combinations at or above the threshold.

	Returns:
		A DataFrame with one row per combination, indexed by the column names joined with ' + ':
			rows used: rows where all of the combination's columns and the target are present.
				The combination and its subsets are all scored on these rows.
			baseline features: the `size - 1` subset with the highest Pearson R.
			added feature: the one column of the combination that is not in the baseline.
			baseline / all features Pearson R: Pearson r of the target with the merged baseline
				columns, and with all the merged columns.
			Pearson R gain: all features Pearson R minus baseline Pearson R.
			baseline / all features Spearman rho: the same, measured with Spearman rho.
			Spearman rho gain: all features Spearman rho minus baseline Spearman rho.
			cohen f²: the gain as Cohen's f².
			cohen f² ≥ threshold, outcome: whether f² meets the threshold, and the suggested outcome.
		Sorted by cohen f², highest first, and rounded to 3 decimals. The threshold is checked
		against the rounded f², so the flag matches what is shown.
	"""
	df = df.drop(columns=target.name, errors='ignore')
	if not 2 <= size <= df.shape[1]:
		raise ValueError(f'size must be between 2 and {df.shape[1]} (the number of columns), got {size}')

	rows = {}
	for features in combinations(df.columns, size):
		complete = pd.concat([df[list(features)], target], axis=1).dropna()
		X = complete[list(features)]
		y = complete[target.name]

		combined_r, combined_rho = _fitted_correlations(X, y)

		# every way to leave one column out, scored on the same rows as the full combination
		subsets = {
			subset: _fitted_correlations(X[list(subset)], y)
			for subset in combinations(features, size - 1)
		}
		baseline = max(subsets, key=lambda subset: subsets[subset][0])
		baseline_r, baseline_rho = subsets[baseline]
		(added_feature,) = set(features) - set(baseline)

		rows[' + '.join(features)] = {
			'rows used': len(complete),
			'baseline features': ' + '.join(baseline),
			'added feature': added_feature,
			'baseline Pearson R': baseline_r,
			'all features Pearson R': combined_r,
			'Pearson R gain': combined_r - baseline_r,
			'baseline Spearman rho': baseline_rho,
			'all features Spearman rho': combined_rho,
			'Spearman rho gain': combined_rho - baseline_rho,
			'cohen f²': (combined_r**2 - baseline_r**2) / (1 - combined_r**2),
		}

	out = pd.DataFrame.from_dict(rows, orient='index').round(3)

	meets_cohen_f2 = out['cohen f²'] >= cohen_f2_threshold
	out[f'cohen f² ≥ {cohen_f2_threshold}'] = meets_cohen_f2
	out['outcome'] = np.where(meets_cohen_f2, 'keep together', 'baseline is enough')

	if filter:
		out = out[meets_cohen_f2]
	return out.sort_values('cohen f²', ascending=False)

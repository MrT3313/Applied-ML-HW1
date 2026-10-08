import numpy as np
import pandas as pd

from Q1.utils.statistics.pcc_strength import pcc_strength


def correlation_df(*, data: pd.Series | pd.DataFrame, target: pd.Series) -> pd.DataFrame:
	"""Compute the correlation of one or more columns with the target.

	Args:
		data: A single column (Series) or several columns (DataFrame) to correlate.
			A column named like the target is dropped.
		target: Column every data column is correlated against.

	Returns:
		A DataFrame with one row per column, indexed by column name: Pearson r,
		Spearman rho, and the direction and strength of r. Sorted by |r|, strongest
		first, and rounded to 3 decimals. Columns where r is undefined (ex a constant
		column) get NaN direction and strength and sort last.
	"""
	df = data.to_frame() if isinstance(data, pd.Series) else data
	df = df.drop(columns=target.name, errors='ignore')

	pearson = df.corrwith(target)

	out = pd.DataFrame({
		'Pearson r': pearson,
		'Spearman rho': df.corrwith(target, method='spearman'),
		'direction': np.sign(pearson).map({1.0: 'positive', -1.0: 'negative', 0.0: 'none'}),
		# pcc_strength raises on NaN, so label only defined r; the rest align to NaN
		'strength': pearson.dropna().map(pcc_strength),
	})

	return out.sort_values('Pearson r', key=abs, ascending=False).round(3)

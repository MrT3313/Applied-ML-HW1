from typing import Optional

import pandas as pd


def results_df(*, runs: dict[str, pd.DataFrame], decimals: int = 4, baseline: Optional[str] = None) -> pd.DataFrame:
	"""Summarize cross validation results, one row per result, in the order given.

	Args:
		runs: Results from PIPELINES.cv_rmse keyed by name.
		decimals: Decimals to round to.
		baseline: The name of the run the others are compared with, when there is one; it adds the "% vs baseline"
			column.

	Returns:
		A DataFrame indexed by name:
			validation rmse: Mean validation RMSE over the folds, the score every choice is made on.
			train rmse: Mean training RMSE over the folds. Far below validation rmse means the model fits its training
				rows better than new rows.
			most epochs: The most epochs the model ran on any fold, when it is an SGD model.
			% vs baseline: How much higher (+) or lower (-) the mean validation RMSE is than the baseline's, as a percent
				of the baseline's, when a baseline is given.
	"""
	rows = {}
	for name, run in runs.items():
		row = {'validation rmse': run['validation rmse'].mean(), 'train rmse': run['train rmse'].mean()}
		if 'epochs' in run:
			row['most epochs'] = run['epochs'].max()
		rows[name] = row

	out = pd.DataFrame.from_dict(rows, orient='index')
	if baseline is not None:
		change = 100 * (out['validation rmse'] / out.loc[baseline, 'validation rmse'] - 1)
		# as text, so it keeps 2 decimals whatever the notebook's float format; a diverging run would print a huge number
		out['% vs baseline'] = [
			'baseline' if name == baseline else '> +999%' if value > 999 else f'{value:+.2f}%'
			for name, value in change.items()
		]

	out = out.round(decimals)
	if 'most epochs' in out:
		out['most epochs'] = out['most epochs'].astype('Int64')  # whole numbers, blank on the rows of models without epochs
	return out

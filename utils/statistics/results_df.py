import numpy as np
import pandas as pd

from utils.pipelines.cv_rmse import Folds
from utils.statistics.fold_difference import fold_difference


def results_df(*, runs: dict[str, pd.DataFrame], baseline: str, folds: Folds, decimals: int = 4) -> pd.DataFrame:
	"""Summarize cross validation results against a baseline, one row per result, in the order given.

	Args:
		runs: Results from PIPELINES.cv_rmse keyed by name, all on the same folds. The baseline must be one of them.
		baseline: Name of the result the others are compared with.
		folds: The folds every result was scored on, as row positions.
		decimals: Decimals to round to.

	Returns:
		A DataFrame indexed by name:
			rmse: Mean validation RMSE over the folds.
			train rmse: Mean training RMSE over the folds. Far below rmse means the model fits its training rows
				better than new rows.
			Δrmse: Mean change in validation RMSE against the baseline (below 0 is better).
			SE: Corrected standard error of Δrmse (see STATISTICS.fold_difference).
			better folds: On how many folds the result beats the baseline.
			verdict: 'candidate' (Δrmse below -1 SE), 'within noise', or 'worse' (above +1 SE). 'baseline' on its row.
			most epochs: The most epochs the model ran on any fold, when it is an SGD model.
	"""
	rows = {}
	for name, run in runs.items():
		row = {'rmse': run['validation rmse'].mean(), 'train rmse': run['train rmse'].mean()}
		if name == baseline:
			row.update({'Δrmse': np.nan, 'SE': np.nan, 'better folds': '', 'verdict': 'baseline'})
		else:
			difference = fold_difference(baseline=runs[baseline], candidate=run, folds=folds)
			row.update({
				'Δrmse': difference.mean,
				'SE': difference.se,
				'better folds': f'{difference.better} of {len(difference.diff)}',
				'verdict': difference.verdict,
			})
		if 'epochs' in run:
			row['most epochs'] = run['epochs'].max()
		rows[name] = row

	out = pd.DataFrame.from_dict(rows, orient='index').round(decimals)
	if 'most epochs' in out:
		out['most epochs'] = out['most epochs'].astype('Int64')  # whole numbers, blank on the rows of models without epochs
	return out

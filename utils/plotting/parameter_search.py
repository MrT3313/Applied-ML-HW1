from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consts import BASELINE_COLOR, TRAIN_COLOR, VALIDATION_COLOR


def parameter_search(
	*,
	runs: dict,
	parameter: str,
	log_x: bool = False,
	reference: Optional[tuple[str, pd.DataFrame]] = None,
	ymax: Optional[float] = None,
	title: Optional[str] = None,
) -> plt.Axes:
	"""Plot the RMSE of every value tried for one parameter, fold by fold.

	A thin line per fold joins its validation RMSE across the values, so a value that only wins on some folds shows.
	The thick line is the mean validation RMSE, the dashed line the mean training RMSE, and the star marks the value
	with the lowest mean validation RMSE.

	Args:
		runs: Results from PIPELINES.cv_rmse keyed by the parameter's value (a number), in the order tried.
		parameter: Name of the parameter, for the x axis.
		log_x: Put the values on a log scale, for values spread over several powers of 10.
		reference: (name, result) of a model to draw as a dotted line at its mean validation RMSE, ex the baseline.
		ymax: Highest RMSE shown. A run that diverges can reach RMSEs that flatten every other value against the axis:
			anything above ymax is drawn at ymax as a triangle. None shows everything.
		title: Title of the chart. When omitted, it names the parameter.

	Returns:
		The Axes the chart was drawn on.
	"""
	_, ax = plt.subplots(figsize=(9, 4.5))

	values = list(runs)
	x = np.array(values, dtype=float)

	validation = pd.DataFrame({i: runs[value]['validation rmse'].to_numpy() for i, value in enumerate(values)})  # a row per fold
	mean = validation.mean()
	train_mean = pd.Series([runs[value]['train rmse'].mean() for value in values])

	def shown(*, scores: pd.Series) -> pd.Series:
		return scores.clip(upper=ymax)  # no change when ymax is None

	for i, (_, fold) in enumerate(validation.iterrows()):
		ax.plot(x, shown(scores=fold), '-o', color=VALIDATION_COLOR, alpha=0.25, linewidth=0.8, markersize=3, label='validation, each fold' if i == 0 else None)
	ax.plot(x, shown(scores=mean), '-o', color=VALIDATION_COLOR, linewidth=2.5, label='validation, mean')
	ax.plot(x, shown(scores=train_mean), '--x', color=TRAIN_COLOR, linewidth=2, label='train, mean')

	best = int(mean.idxmin())
	ax.plot(x[best], mean[best], '*', color='black', markersize=14, label=f'lowest mean: {values[best]}')

	if ymax is not None:
		above = [(x[i], ymax) for i in validation.columns for score in validation[i] if score > ymax]
		if above:
			ax.plot(*zip(*above), '^', color='black', markersize=7, label=f'above {ymax:g} (off the chart)')
	if reference is not None:
		name, run = reference
		ax.axhline(run['validation rmse'].mean(), color=BASELINE_COLOR, linestyle=':', linewidth=1.5, label=f'{name}, validation mean')

	if log_x:
		ax.set_xscale('log')
	else:
		ax.set_xticks(x)
	ax.set_xlabel(parameter + (' (log scale)' if log_x else ''))
	ax.set_ylabel('RMSE')
	ax.set_title(title or f'{parameter}: RMSE on every fold')
	ax.grid(alpha=0.3)
	ax.legend(fontsize='small')

	return ax

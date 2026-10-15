from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from consts import LOSS_METHOD_COLORS, TRAIN_COLOR, VALIDATION_COLOR


def loss_curve(*, curve: pd.DataFrame, title: str = 'RMSE by epoch', log_x: bool = False, ymax: Optional[float] = None) -> plt.Axes:
	"""Plot training and validation RMSE against the epoch: a thin line per fold and a thick line for their mean.

	Args:
		curve: Per fold, per epoch scores from PIPELINES.epoch_rmse.
		title: Title of the chart.
		log_x: Put the epochs on a log scale, so the first epochs stay readable next to a long run.
		ymax: Highest RMSE shown when a run goes above it, so a run that diverges leaves the top of the chart instead of
			flattening the rest. None shows everything.

	Returns:
		The Axes the chart was drawn on.
	"""
	_, ax = plt.subplots(figsize=(9, 4.5))
	mean = curve.groupby(level='epoch').mean()
	n_folds = curve.index.get_level_values('fold').nunique()

	for column, color, label in (('train rmse', TRAIN_COLOR, 'train'), ('validation rmse', VALIDATION_COLOR, 'validation')):
		for _, fold in curve[column].groupby(level='fold'):
			ax.plot(fold.index.get_level_values('epoch'), fold, color=color, alpha=0.3, linewidth=0.8)
		ax.plot(mean.index, mean[column], color=color, linewidth=2.5, label=f'{label} (thin: each of {n_folds} folds, thick: mean)')

	if log_x:
		ax.set_xscale('log')
	else:
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
	ax.set_xlabel('Epoch' + (' (log scale)' if log_x else ''))
	ax.set_ylabel('RMSE')
	if ymax is not None and curve.to_numpy().max() > ymax:
		ax.set_ylim(0.95 * curve.to_numpy().min(), ymax)  # only when a run goes above ymax; the others keep their own scale
	ax.set_title(title)
	ax.grid(alpha=0.3)
	ax.legend()

	return ax


def loss_curve_comparison(
	*,
	curves: dict[str, pd.DataFrame],
	title: str = 'RMSE by epoch, mean of the folds',
	ymax: Optional[float] = None,
) -> plt.Axes:
	"""Plot the mean training and validation RMSE of two loss curves of one model on one chart, so they can be read
	against each other: a color per curve, solid for validation and dashed for training.

	Args:
		curves: Per fold, per epoch scores from PIPELINES.epoch_rmse keyed by method ('warm_start' or 'refit'), the
			order they are drawn and listed in.
		title: Title of the chart.
		ymax: Highest RMSE shown when a curve goes above it, as in loss_curve. None shows everything.

	Returns:
		The Axes the chart was drawn on.
	"""
	labels = {'warm_start': "the homework's loop", 'refit': 'exact'}
	_, ax = plt.subplots(figsize=(9, 4.5))
	highest, lowest = float('-inf'), float('inf')
	for method, curve in curves.items():
		mean = curve.groupby(level='epoch').mean()
		highest, lowest = max(highest, mean.to_numpy().max()), min(lowest, mean.to_numpy().min())
		color = LOSS_METHOD_COLORS[method]
		ax.plot(mean.index, mean['validation rmse'], color=color, linewidth=2.5, label=f'{labels[method]}: validation')
		ax.plot(mean.index, mean['train rmse'], color=color, linewidth=1.5, linestyle='--', label=f'{labels[method]}: train')

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
	ax.set_xlabel('Epoch')
	ax.set_ylabel('RMSE, mean of the folds')
	if ymax is not None and highest > ymax:
		ax.set_ylim(0.95 * lowest, ymax)
	ax.set_title(title)
	ax.grid(alpha=0.3)
	ax.legend()

	return ax

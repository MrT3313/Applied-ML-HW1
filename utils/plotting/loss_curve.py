import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from consts import TRAIN_COLOR, VALIDATION_COLOR


def loss_curve(*, curve: pd.DataFrame, title: str = 'RMSE by epoch') -> plt.Axes:
	"""Plot training and validation RMSE against the epoch: a thin line per fold and a thick line for their mean.

	Args:
		curve: Per fold, per epoch scores from PIPELINES.epoch_rmse.
		title: Title of the chart.

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

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
	ax.set_xlabel('Epoch')
	ax.set_ylabel('RMSE')
	ax.set_title(title)
	ax.grid(alpha=0.3)
	ax.legend()

	return ax

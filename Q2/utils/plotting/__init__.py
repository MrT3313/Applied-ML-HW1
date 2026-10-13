import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Q2.utils.plotting.level_counts import level_counts
from Q2.utils.plotting.level_box_plot import level_box_plot
from Q2.utils.plotting.predictions_by_date import predictions_by_date

def categorical_feature_plots(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
) -> tuple[plt.Figure, np.ndarray]:
	"""Plot one categorical feature's row count per level and a box plot of the target for each level.

	Args:
		df: DataFrame holding both columns.
		feature: Categorical column to plot.
		target: Column whose distribution is plotted for each level.

	Returns:
		The Figure and its array of 2 Axes ([row counts, box plot]).
	"""
	fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

	level_counts(series=df[feature], ax=axes[0])
	level_box_plot(df=df, feature=feature, target=target, ax=axes[1])

	fig.tight_layout()

	return fig, axes

__all__ = [
	'categorical_feature_plots',

	'level_counts',
	'level_box_plot',
	'predictions_by_date',
]

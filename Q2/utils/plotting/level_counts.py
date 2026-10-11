from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def level_counts(
	*,
	series: pd.Series,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a bar chart of the number of rows at each level of a categorical column.

	Args:
		series: Categorical values to count. Missing values are dropped.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	counts = series.dropna().value_counts().sort_index()

	if ax is None:
		_, ax = plt.subplots()

	# levels are drawn as text labels so codes are spaced evenly instead of on a numeric scale
	bars = ax.bar([str(level) for level in counts.index], counts.to_numpy())
	ax.bar_label(bars)  # row count above each bar

	ax.set_title(f'{series.name} Rows per Level')
	ax.set_xlabel(series.name)
	ax.set_ylabel('Rows')

	return ax

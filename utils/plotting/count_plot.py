from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from consts import DEFAULT_BLUE, GROUP_COLORS


def count_plot(
	*,
	df: pd.DataFrame,
	feature: str,
	by: Optional[str] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a bar chart of the number of rows at each level of a categorical column.

	Args:
		df: DataFrame holding the columns. Rows missing any of them are dropped.
		feature: Categorical column whose levels are spread along the x axis.
		by: Second categorical column that splits each level into side by side bars, one color per level of it.
			When omitted, each level is one bar.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	if by:
		counts = pd.crosstab(df[feature], df[by])  # rows: feature levels, columns: by levels; 0 where a pair has no rows
		counts.columns = [f'{by} {level}' for level in counts.columns]  # legend labels
		color = GROUP_COLORS[:len(counts.columns)]
	else:
		counts = df[feature].value_counts().sort_index()
		color = DEFAULT_BLUE

	if ax is None:
		_, ax = plt.subplots()

	# pandas draws the levels as evenly spaced text labels, so codes are not placed on a numeric scale
	counts.plot.bar(ax=ax, color=color, width=0.8, rot=0, legend=False)
	for bars in ax.containers:
		ax.bar_label(bars, fontsize='small')  # row count above each bar, 0 included

	ax.set_title(f'{feature} Rows per Level' + (f' by {by}' if by else ''))
	ax.set_xlabel(feature)
	ax.set_ylabel('Rows')

	if by:
		# Legend centered below the axes, outside the plot area
		ax.legend(
			loc='upper center',  # anchor point on the legend box
			bbox_to_anchor=(0.5, -0.15),  # place that anchor centered, just below the x-axis label
			ncol=len(counts.columns),
			fontsize='small',
			framealpha=0.9,
		)

	return ax

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def grouped_level_counts(
	*,
	df: pd.DataFrame,
	feature: str,
	by: str,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a grouped bar chart of the number of rows at each level of a categorical column, with one bar per level of a second column.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Categorical column whose levels are spread along the x axis.
		by: Categorical column that splits each level into side by side bars, one color per level of it.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	counts = pd.crosstab(df[feature], df[by])  # rows: feature levels, columns: by levels; 0 where a pair has no rows

	if ax is None:
		_, ax = plt.subplots()

	# each feature level is centered on x = 1, 2, 3, ... and its bars share 0.8 of the space to the next level
	bar_width = 0.8 / len(counts.columns)
	for i, by_level in enumerate(counts.columns):
		offset = (i - (len(counts.columns) - 1) / 2) * bar_width
		bars = ax.bar(
			[x + offset for x in range(1, len(counts) + 1)],
			counts[by_level].to_numpy(),
			width=bar_width,
			color=f'C{i}',  # the same color as this by level's boxes in grouped_level_box_plot
			label=f'{by} {by_level}',
		)
		ax.bar_label(bars, fontsize='small')  # row count above each bar, 0 included

	ax.set_xticks(range(1, len(counts) + 1), labels=[str(level) for level in counts.index])

	ax.set_title(f'{feature} Rows per Level by {by}')
	ax.set_xlabel(feature)
	ax.set_ylabel('Rows')

	# Legend centered below the axes, outside the plot area
	ax.legend(
		loc='upper center',  # anchor point on the legend box
		bbox_to_anchor=(0.5, -0.15),  # place that anchor centered, just below the x-axis label
		ncol=len(counts.columns),
		fontsize='small',
		framealpha=0.9,
	)

	return ax

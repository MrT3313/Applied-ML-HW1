from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from utils.plotting.box_plot import draw_boxes


def grouped_level_box_plot(
	*,
	df: pd.DataFrame,
	feature: str,
	by: str,
	target: str,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a box plot of the target for each level of a categorical feature, with one box per level of a second column.

	A level with no rows at one of the second column's levels has no box there.

	Args:
		df: DataFrame holding the three columns. Rows missing any of them are dropped.
		feature: Categorical column whose levels are spread along the x axis.
		by: Categorical column that splits each level into side by side boxes, one color per level of it.
		target: Column whose distribution is plotted in each box.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	data = df[[feature, by, target]].dropna()  # boxplot returns NaN stats for a group containing NaN
	levels = sorted(data[feature].unique())
	by_levels = sorted(data[by].unique())

	if ax is None:
		_, ax = plt.subplots()

	# each feature level is centered on x = 1, 2, 3, ... and its boxes share 0.8 of the space to the next level
	slot_width = 0.8 / len(by_levels)
	legend_handles = []
	for i, by_level in enumerate(by_levels):
		offset = (i - (len(by_levels) - 1) / 2) * slot_width
		color = to_rgba(f'C{i}', 0.6)  # the same color as this by level's bars in grouped_level_counts, lightened so the median and mean show

		groups, positions = {}, []
		for x, level in enumerate(levels, start=1):
			values = data.loc[(data[feature] == level) & (data[by] == by_level), target]
			if len(values) > 0:  # no box where the level has no rows at this by level
				groups[f'{level} / {by_level}'] = values
				positions.append(x + offset)

		draw_boxes(groups=groups, ax=ax, mean_label=None, positions=positions, width=slot_width * 0.85, color=color)
		legend_handles.append(Patch(facecolor=color, edgecolor='black', label=f'{by} {by_level}'))

	ax.plot([], [], 'D', color='red', markersize=5, label='Mean')  # legend entry for the mean markers
	legend_handles += ax.get_legend_handles_labels()[0]

	ax.set_xticks(range(1, len(levels) + 1), labels=[str(level) for level in levels])
	ax.set_xlim(0.5, len(levels) + 0.5)  # the same span as grouped_level_counts, with room for a missing box at either end

	ax.set_xlabel(feature)
	ax.set_ylabel(target)
	ax.set_title(f'{target} by {feature} and {by} Box Plot')

	# Legend centered below the axes, outside the plot area
	ax.legend(
		handles=legend_handles,
		loc='upper center',  # anchor point on the legend box
		bbox_to_anchor=(0.5, -0.15),  # place that anchor centered, just below the x-axis label
		ncol=len(legend_handles),
		fontsize='small',
		framealpha=0.9,
	)

	return ax

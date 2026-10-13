from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from utils.plotting.box_plot import draw_boxes
from Q2.utils.statistics.category_levels_df import level_means_r_squared


def level_box_plot(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a box plot of the target for each level of a categorical feature, with the overall target mean as a reference line.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Categorical column that splits the rows into one group per level.
		target: Column whose distribution is plotted for each level.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	data = df[[feature, target]].dropna()  # boxplot returns NaN stats for a group containing NaN
	groups = {level: group[target] for level, group in data.groupby(feature)}
	r_squared = level_means_r_squared(df=data, feature=feature, target=target)

	if ax is None:
		_, ax = plt.subplots()

	overall_mean = data[target].mean()
	ax.axhline(overall_mean, color='gray', ls='--', lw=1.5, label=f'Overall mean = {overall_mean:,.0f}')

	draw_boxes(
		groups={f'{level}\nn={len(values)}': values for level, values in groups.items()},  # labels kept short so 12 levels fit side by side
		ax=ax,
		mean_label='Level mean',  # each level's mean is what the level means R² is built from
	)

	ax.set_xlabel(feature)
	ax.set_ylabel(target)
	ax.set_title(f'{target} by {feature} Box Plot (level means R² = {r_squared:.2f})')

	# Legend centered below the axes, outside the plot area
	ax.legend(
		loc='upper center',  # anchor point on the legend box
		bbox_to_anchor=(0.5, -0.2),  # place that anchor centered, just below the x-axis label
		ncol=2,
		fontsize='small',
		framealpha=0.9,
	)

	return ax

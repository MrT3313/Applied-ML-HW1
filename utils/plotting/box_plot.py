from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from consts import DEFAULT_BLUE, HINGE_COLOR, MEAN_COLOR, MEDIAN_COLOR, STEP_COLOR

# boxes split at a step or hinge threshold take the color of that fit's line on the scatter plot
THRESHOLD_COLORS = {'step': STEP_COLOR, 'hinge': HINGE_COLOR}


def draw_boxes(
	*,
	groups: dict[str, pd.Series],
	ax: plt.Axes,
	vertical: bool = False,
	colors: Optional[Sequence[str]] = None,
) -> dict:
	"""Draw one box per group, each with its median line and mean marker.

	Every box plot goes through here, so they all share one look. The caller draws the legend.

	Args:
		groups: Values of each box keyed by its tick label, in the order drawn. Must not contain NaN.
		ax: Axes to draw on.
		vertical: Draw the boxes upright, side by side from left to right. When omitted, they lie flat, stacked from top to bottom.
		colors: Fill color of each box, in the order drawn. Every box is the default blue when omitted.

	Returns:
		The artists Axes.boxplot drew, keyed by part ('means', 'medians', ...).
	"""
	parts = ax.boxplot(
		list(groups.values()),
		vert=vertical,
		widths=0.5,
		patch_artist=True,  # draw each box as a filled patch so it can take a face color
		medianprops={'color': MEDIAN_COLOR, 'lw': 2},
		showmeans=True,
		meanprops={'marker': 'D', 'markerfacecolor': MEAN_COLOR, 'markeredgecolor': 'white', 'markersize': 7},
		flierprops={'marker': 'o', 'markersize': 3},
	)
	# each box and its outlier points share one color
	for box, fliers, color in zip(parts['boxes'], parts['fliers'], colors or [DEFAULT_BLUE] * len(groups)):
		box.set_facecolor(color)
		fliers.set(markerfacecolor=color, markeredgecolor=color)

	ticks = range(1, len(groups) + 1)  # boxplot places the boxes at 1, 2, 3, ...
	if vertical:
		ax.set_xticks(ticks, labels=list(groups))
	else:
		ax.set_yticks(ticks, labels=list(groups))
		ax.invert_yaxis()  # put the first group on top

	return parts


def box_plot(
	*,
	series: pd.Series,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a horizontal box plot of the series with its median line and mean marker.

	Args:
		series: Numeric values to plot. Missing values are dropped.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	x = series.dropna()  # boxplot returns NaN stats for data containing NaN

	if ax is None:
		_, ax = plt.subplots()

	parts = draw_boxes(groups={series.name: x}, ax=ax)
	parts['means'][0].set_label(f'Mean ({x.mean():.2f})')
	parts['medians'][0].set_label(f'Median ({x.median():.2f})')

	ax.set_title(f'{series.name} Box Plot')
	ax.set_xlabel(series.name)
	ax.set_yticks([])  # a single box needs no y tick, the title names it
	ax.legend(
		handles=[parts['means'][0], parts['medians'][0]],  # mean first, the same order as the histogram legend
		loc='upper right',
		fontsize='small',
		framealpha=0.9,
	)

	return ax


def box_plot_vs_target(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	thresholds: Optional[dict[str, float]] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a vertical box plot of the target for all rows and for the two groups each feature threshold splits them into.

	Args:
		df: DataFrame holding both columns. Rows missing either one are dropped.
		feature: Column used to split the rows into groups.
		target: Column whose distribution is plotted for each group.
		thresholds: Feature value of each split keyed by its name, such as 'step' or 'hinge'. Every split adds two boxes:
			rows with feature < threshold, then the rest. When omitted, only the box of all rows is drawn.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	data = df[[feature, target]].dropna()  # boxplot returns NaN stats for a group containing NaN
	groups = {'all': data[target]}
	colors = [DEFAULT_BLUE]
	for name, threshold in (thresholds or {}).items():
		groups[f'{name}\n< {threshold:g}'] = data.loc[data[feature] < threshold, target]
		groups[f'{name}\n>= {threshold:g}'] = data.loc[data[feature] >= threshold, target]
		colors += [THRESHOLD_COLORS.get(name, DEFAULT_BLUE)] * 2  # one color for both sides of the split

	if ax is None:
		_, ax = plt.subplots()

	parts = draw_boxes(
		groups={f'{label}\n(n = {len(values)})': values for label, values in groups.items()},
		ax=ax,
		vertical=True,
		colors=colors,
	)
	parts['means'][0].set_label('Mean')
	parts['medians'][0].set_label('Median')

	ax.set_title(f'{target} vs {feature} Box Plot')
	ax.set_xlabel(feature)  # the thresholds in the tick labels are values of the feature
	ax.set_ylabel(target)

	# Legend centered below the axes, outside the plot area
	ax.legend(
		handles=[parts['means'][0], parts['medians'][0]],  # mean first, the same order as the histogram legend
		loc='upper center',  # anchor point on the legend box
		bbox_to_anchor=(0.5, -0.25),  # place that anchor centered, just below the x-axis label
		ncol=2,
		fontsize='small',
		framealpha=0.9,
	)

	return ax

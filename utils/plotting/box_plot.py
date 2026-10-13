from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def draw_boxes(
	*,
	groups: dict[str, pd.Series],
	ax: plt.Axes,
	mean_label: Optional[str] = 'Mean',
	positions: Optional[Sequence[float]] = None,
	width: Optional[float] = None,
	color: Optional[str] = None,
) -> None:
	"""Draw one box per group, each with its mean marked by a red diamond.

	Every box plot goes through here, so they all share one look.

	Args:
		groups: Values of each box keyed by its x tick label, in the order drawn. Must not contain NaN.
		ax: Axes to draw on.
		mean_label: Legend entry for the mean markers, or None for no entry. The caller draws the legend.
		positions: x position of each box. When omitted, the boxes sit at x = 1, 2, 3, ... and are labeled with
			the keys of groups. When given, the x ticks are left to the caller.
		width: Width of each box. matplotlib picks one when omitted.
		color: Fill color of the boxes, used to tell sets of boxes apart. The boxes are unfilled when omitted.
	"""
	filled = {} if color is None else {
		'patch_artist': True,
		'boxprops': {'facecolor': color},
		'medianprops': {'color': 'black'},  # the default orange median is hard to see on a fill
	}

	ax.boxplot(
		list(groups.values()),
		positions=positions,
		widths=width,
		manage_ticks=positions is None,
		showmeans=True,
		meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 5},
		**filled,
	)

	if positions is None:
		ax.set_xticks(range(1, len(groups) + 1), labels=list(groups))  # boxplot places the boxes at x = 1, 2, 3, ...
	if mean_label is not None:
		ax.plot([], [], 'D', color='red', markersize=5, label=mean_label)  # legend entry for the mean markers


def box_plot(
	*,
	df: pd.DataFrame,
	feature: str,
	target: Optional[str] = None,
	threshold: Optional[float] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a single box plot of the feature, or box plots of the target for all rows and for two groups split by a feature threshold.

	Each box has its mean marked by a red diamond.

	Args:
		df: DataFrame holding both columns. Missing values in the plotted columns are dropped.
		feature: Column plotted on its own when threshold is omitted, otherwise used to split the rows into groups.
		target: Column whose distribution is plotted for each group. Required when threshold is given, unused otherwise.
		threshold: Rows with feature < threshold form the low group; the rest form the high group.
			The target over all rows is drawn as a third box on the far left.
			When omitted, a single box of the feature is drawn instead.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	if ax is None:
		_, ax = plt.subplots()

	if threshold is None:
		values = df[feature].dropna()  # boxplot returns NaN stats for data containing NaN
		groups = {f'{feature}\n(n = {len(values)})': values}

		ax.set_ylabel(feature)
		ax.set_title(f'{feature} Box Plot')
	else:
		data = df[[feature, target]].dropna()  # boxplot returns NaN stats for a group containing NaN

		full_group = data[target]
		low_group = data.loc[data[feature] < threshold, target]
		high_group = data.loc[data[feature] >= threshold, target]
		groups = {
			f'all\n(n = {len(full_group)})': full_group,
			f'{feature} < {threshold:g}\n(n = {len(low_group)})': low_group,
			f'{feature} >= {threshold:g}\n(n = {len(high_group)})': high_group,
		}

		ax.set_ylabel(target)
		ax.set_title(f'{target} vs {feature} Box Plot')

	draw_boxes(groups=groups, ax=ax)

	# Legend centered below the axes, outside the plot area
	ax.legend(
		loc='upper center',  # anchor point on the legend box
		bbox_to_anchor=(0.5, -0.15),  # place that anchor centered, just below the x tick labels
		fontsize='small',
		framealpha=0.9,
	)

	return ax

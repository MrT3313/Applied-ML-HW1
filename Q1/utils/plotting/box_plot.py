from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from Q1.consts import TARGET


def box_plot(
	*,
	df: pd.DataFrame,
	feature: str,
	target: Optional[str] = TARGET,
	threshold: Optional[float] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a single box plot of the feature, or box plots of the target for all rows and for two groups split by a feature threshold.

	Args:
		df: DataFrame holding both columns. Missing values in the plotted columns are dropped.
		feature: Column plotted on its own when threshold is omitted, otherwise used to split the rows into groups.
		target: Column whose distribution is plotted for each group. Unused when threshold is omitted.
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

		ax.boxplot(values)
		ax.set_xticks([1], labels=[f'{feature}\n(n = {len(values)})'])  # boxplot places the box at x = 1

		ax.set_ylabel(feature)
		ax.set_title(f'{feature} Box Plot')

		return ax

	data = df[[feature, target]].dropna()  # boxplot returns NaN stats for a group containing NaN

	full_group = data[target]
	low_group = data.loc[data[feature] < threshold, target]
	high_group = data.loc[data[feature] >= threshold, target]

	ax.boxplot([full_group, low_group, high_group])
	ax.set_xticks([1, 2, 3], labels=[  # boxplot places the boxes at x = 1, 2, 3
		f'all\n(n = {len(full_group)})',
		f'{feature} < {threshold:g}\n(n = {len(low_group)})',
		f'{feature} >= {threshold:g}\n(n = {len(high_group)})',
	])

	ax.set_ylabel(target)
	ax.set_title(f'{target} vs {feature} Box Plot')

	return ax

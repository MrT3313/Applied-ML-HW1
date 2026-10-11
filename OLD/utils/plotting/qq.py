from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def qq(
	*,
	series: pd.Series,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a normal Q-Q plot of the series against a fitted reference line.

	Args:
		series: Numeric values to plot. Missing values are dropped.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	if ax is None:
		_, ax = plt.subplots()

	stats.probplot(
		series.dropna(),
		dist="norm",
		plot=ax)
	ax.set_title(f'{series.name} QQ Plot')  # probplot titles every plot "Probability Plot" by default
	ax.get_lines()[0].set_markersize(3)  # data points; get_lines()[1] is the red reference line

	return ax

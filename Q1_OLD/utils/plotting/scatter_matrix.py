from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix as pd_scatter_matrix

from Q1_OLD.consts import SHORT_NAMES


def scatter_matrix(
	*,
	df: pd.DataFrame,
	labels: Optional[dict[str, str]] = SHORT_NAMES,
	diagonal: Literal["kde", "hist"] = "kde",
	title: Optional[str] = None,
	figsize: tuple[float, float] = (10, 10),
	alpha: float = 0.3,
	s: float = 8,
) -> np.ndarray:
	"""Plot a scatter of every pair of numeric columns, with each column's distribution on the diagonal.

	Args:
		df: DataFrame to plot. Non-numeric columns are skipped, and missing values are dropped per pair.
		labels: Maps column names to the shorter axis labels shown on the grid. Columns not in the map keep
			their own name. When omitted, the full column names are used.
		diagonal: Draw each column's distribution on the diagonal as a KDE curve or a histogram.
		title: Figure title drawn above the grid.
		figsize: Size of the whole figure in inches.
		alpha: Opacity of the scatter points.
		s: Size of the scatter points.

	Returns:
		The 2D array of Axes, one per column pair. The Figure is axes[0, 0].figure.
	"""
	plot_df = df.rename(columns=labels) if labels else df

	axes = pd_scatter_matrix(plot_df, figsize=figsize, alpha=alpha, s=s, diagonal=diagonal)

	# only the outer row and column show labels; rotating them keeps long names from overlapping their neighbors
	for ax in axes.flatten():
		ax.xaxis.label.set_rotation(45)
		ax.yaxis.label.set_rotation(0)
		ax.yaxis.label.set_ha('right')  # horizontal y labels sit left of the axis instead of centered over it

	fig = axes[0, 0].figure

	if title:
		fig.suptitle(title, y=1.01)  # just above the top row, which tight_layout would otherwise crowd

	fig.tight_layout()

	return axes

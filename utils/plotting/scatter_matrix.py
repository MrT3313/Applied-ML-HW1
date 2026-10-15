from typing import Optional

import pandas as pd
import seaborn as sns

FIGSIZE = (10, 10)  # inches, for the whole grid however many columns it has


def scatter_matrix(
	*,
	df: pd.DataFrame,
	labels: Optional[dict[str, str]] = None,
	title: Optional[str] = None,
) -> sns.PairGrid:
	"""Plot a scatter of every pair of numeric columns, with each column's KDE curve on the diagonal.

	Args:
		df: DataFrame to plot. Non-numeric columns are skipped, and missing values are dropped per pair.
		labels: Maps column names to the shorter axis labels shown on the grid. Columns not in the map keep
			their own name. When omitted, the full column names are used.
		title: Title drawn above the whole grid.

	Returns:
		The seaborn PairGrid. Its Figure is grid.figure and its 2D array of Axes is grid.axes.
	"""
	grid = sns.pairplot(
		df.rename(columns=labels or {}),
		diag_kind='kde',
		plot_kws={'s': 2, 'alpha': 0.3, 'linewidth': 0},  # small, faint, edgeless points so dense areas stay readable
	)

	# seaborn sizes each panel, so the figure would grow with the column count
	grid.figure.set_size_inches(FIGSIZE)

	if title:
		grid.figure.suptitle(title)

	grid.tight_layout()  # after the title, so the grid makes room for it

	return grid

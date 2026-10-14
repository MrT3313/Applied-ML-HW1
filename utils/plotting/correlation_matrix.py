from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Seaborn's rocket and mako, reversed so 0 is light and 1 is near black. Their lightness ramps match, so equal
# strength looks equally dark in both triangles, and hue alone tells them apart: warm for Pearson, cool for Spearman.
PEARSON_CMAP = 'rocket_r'
SPEARMAN_CMAP = 'mako_r'


def correlation_matrix(
	*,
	pearson: pd.DataFrame,
	spearman: pd.DataFrame,
	labels: Optional[dict[str, str]] = None,
	title: Optional[str] = None,
	figsize: tuple[float, float] = (11, 7),
) -> plt.Axes:
	"""Plot Pearson r in the lower left triangle and Spearman rho in the upper right triangle of one annotated heatmap.

	Each cell prints the signed coefficient and is colored by its absolute value, so darker means stronger in
	either direction. Each triangle has its own colormap and colorbar.

	Args:
		pearson: Square Pearson correlation matrix, as returned by DataFrame.corr(). Its diagonal of 1s is drawn.
		spearman: Square Spearman correlation matrix over the same columns, as returned by
			DataFrame.corr(method='spearman').
		labels: Maps column names to the shorter tick labels shown on the heatmap. Columns not in the map keep
			their own name. When omitted, the full column names are used.
		title: Title drawn above the heatmap.
		figsize: Size of the figure in inches.

	Returns:
		The Axes the chart was drawn on.
	"""
	if not pearson.columns.equals(spearman.columns):
		raise ValueError('pearson and spearman must have the same columns in the same order')

	if labels:
		pearson = pearson.rename(index=labels, columns=labels)
		spearman = spearman.rename(index=labels, columns=labels)

	# constrained layout makes room for both colorbars and their labels without overlap
	_, ax = plt.subplots(figsize=figsize, layout='constrained')

	above_diagonal = np.triu(np.ones_like(pearson, dtype=bool), k=1)

	# both layers share these, so the triangles line up cell for cell
	shared = dict(fmt='.2f', vmin=0, vmax=1, square=True, ax=ax)

	# Spearman is drawn first because each new colorbar is placed left of the one before it
	sns.heatmap(
		spearman.abs(),  # color by strength
		annot=spearman,  # print the signed coefficient
		mask=~above_diagonal,  # hide the diagonal and below, which Pearson fills
		cmap=SPEARMAN_CMAP,
		cbar_kws={'label': 'Strength of Spearman rho, upper right (sign ignored)'},
		**shared,
	)
	sns.heatmap(
		pearson.abs(),
		annot=pearson,
		mask=above_diagonal,
		cmap=PEARSON_CMAP,
		cbar_kws={'label': 'Strength of Pearson r, lower left (sign ignored)'},
		**shared,
	)

	ax.tick_params(axis='y', rotation=0)  # seaborn turns the row labels vertical when the rows are tall

	if title:
		ax.set_title(title)

	return ax

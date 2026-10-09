from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from Q1_OLD.consts import SHORT_NAMES
from Q1_OLD.utils.statistics.pcc_strength import PCC_THRESHOLDS

# Seaborn's perceptually uniform rocket and mako, reversed so |r| = 0 is light and |r| = 1 is near black.
# Their lightness ramps match, so equal |r| looks equally dark in both triangles, and hue alone tells them apart:
# warm (cream to red to plum) for Pearson, cool (mint to teal to indigo) for Spearman.
PEARSON_CMAP = 'rocket_r'
SPEARMAN_CMAP = 'mako_r'


def _mirrored(name: str) -> ListedColormap:
	"""Mirror a sequential colormap around 0 so r and -r get the same color, matching vmin = -1 to vmax = 1."""
	base = mpl.colormaps[name]
	# positions run 0 to 1 across r = -1 to 1, so |r| = |2 * position - 1|
	positions = np.linspace(0, 1, 512)
	return ListedColormap(base(np.abs(2 * positions - 1)), name=f'{name}_mirrored')


def correlation_matrix(
	*,
	pearson: pd.DataFrame,
	spearman: pd.DataFrame,
	labels: Optional[dict[str, str]] = SHORT_NAMES,
	title: Optional[str] = None,
	figsize: tuple[float, float] = (11, 7),
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot Pearson r in the lower left triangle and Spearman rho in the upper right triangle of one annotated heatmap.

	Each triangle has its own colormap and colorbar, both light at 0 and near black at |r| = 1 with the same
	lightness in between: warm for Pearson, cool for Spearman. Colorbar ticks mark each PCC_THRESHOLDS cutoff.

	Args:
		pearson: Square Pearson correlation matrix, as returned by DataFrame.corr(). Its diagonal of 1s is drawn.
		spearman: Square Spearman correlation matrix over the same columns, as returned by
			DataFrame.corr(method='spearman').
		labels: Maps column names to the shorter tick labels shown on the heatmap. Columns not in the map keep
			their own name. When omitted, the full column names are used.
		title: Title drawn above the heatmap.
		figsize: Size of the new figure in inches. Unused when ax is given.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	if not pearson.columns.equals(spearman.columns):
		raise ValueError('pearson and spearman must have the same columns in the same order')

	if labels:
		pearson = pearson.rename(index=labels, columns=labels)
		spearman = spearman.rename(index=labels, columns=labels)

	ones = np.ones_like(pearson, dtype=bool)

	# colorbar ticks at each strength threshold on both signs, plus the ends of the scale
	cutoffs = sorted({*PCC_THRESHOLDS.values(), 1.0})  # 0, 0.2, 0.4, 0.7, 1
	ticks = [-t for t in reversed(cutoffs) if t > 0] + cutoffs  # mirror below 0 without repeating 0

	if ax is None:
		_, ax = plt.subplots(figsize=figsize)

	# both layers share these, so the triangles line up cell for cell
	shared = dict(
		annot=True, fmt='.2f',  # print the coefficient in each cell to 2 decimals
		vmin=-1, vmax=1,  # pin the color scale to the full range so the colormap positions line up
		square=True,
		ax=ax,
	)

	sns.heatmap(
		pearson,
		mask=np.triu(ones, k=1),  # hide above the diagonal; k=1 keeps the diagonal of 1s
		cmap=_mirrored(PEARSON_CMAP),
		cbar_kws={
			'label': 'Pearson r, lower left (darker = stronger)',
			'location': 'left',  # beside the lower left triangle it describes; Spearman's colorbar stays on the right
			'pad': 0.18,  # leave room for the row labels between the colorbar and the heatmap
			'ticks': ticks,
		},
		**shared,
	)

	sns.heatmap(
		spearman,
		mask=np.tril(ones),  # hide the diagonal and below, which Pearson already fills
		cmap=_mirrored(SPEARMAN_CMAP),
		cbar_kws={
			'label': 'Spearman rho, upper right (darker = stronger)',
			'ticks': ticks,
		},
		**shared,
	)

	if title:
		ax.set_title(title)

	ax.figure.tight_layout()

	return ax

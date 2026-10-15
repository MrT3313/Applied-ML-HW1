from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from consts import FAMILY_COLORS, LOWESS_COLOR, REFERENCE_COLOR, SCATTER_COLOR
from utils.statistics.shape_fits import ShapeFit, lowess_fit

# each family's curves also get their own line style, so the lines stay distinguishable where they overlap
FAMILY_LINE_STYLES = {
	're-expression': '-.',
	'piecewise linear': '--',
	'bins': ':',
}


def scatterplot(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	fits: Sequence[ShapeFit] = (),
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a scatter of target vs feature with a fitted linear line, a LOWESS curve, and optional candidate shapes.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Column plotted on the x axis.
		target: Column plotted on the y axis.
		fits: Candidate shapes (from shape_fits) to draw, each in its family's color and line style.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	data = df[[feature, target]].dropna()
	x, y = data[feature].to_numpy(dtype=float), data[target].to_numpy(dtype=float)

	if ax is None:
		_, ax = plt.subplots()

	ax.scatter(x, y, color=SCATTER_COLOR, alpha=0.35, s=12, linewidths=0)

	# 1,000 evenly spaced x values used to draw the fits, close enough together that a step or bin edge draws as a jump
	xs = np.linspace(x.min(), x.max(), 1000)

	# slope is in target units per feature unit; r is unitless in [-1, 1]
	fit = stats.linregress(x, y)
	ax.plot(
		xs,
		fit.intercept + fit.slope * xs,
		color=REFERENCE_COLOR,
		lw=2,
		label=f'Linear: slope = {fit.slope:.2f}, r = {fit.rvalue:.2f}, R² = {fit.rvalue**2:.2f}',
	)

	# LOWESS follows the local average without assuming a shape
	fitted, r_squared = lowess_fit(x=x, y=y)
	order = np.argsort(x)  # draw the curve from left to right
	ax.plot(x[order], fitted[order], color=LOWESS_COLOR, lw=2, label=f'LOWESS: R² = {r_squared:.2f}')

	for shape in fits:
		ax.plot(
			xs,
			shape.predict(x=xs),
			color=FAMILY_COLORS[shape.family],
			ls=FAMILY_LINE_STYLES[shape.family],
			lw=2,
			label=f'{shape.label}: R² = {shape.r_squared:.2f}',
		)

	ax.set_title(f'{target} vs {feature} Scatter Plot')
	ax.set_xlabel(feature)
	ax.set_ylabel(target)

	# Legend centered below the axes, outside the plot area
	ax.legend(
		loc='upper center',  # anchor point on the legend box
		bbox_to_anchor=(0.5, -0.15),  # place that anchor centered, just below the x-axis label
		ncol=2,  # entries are long, so 2 columns keeps the legend short without overflowing the axes width
		fontsize='small',
		framealpha=0.9,
	)

	return ax

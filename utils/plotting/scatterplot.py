from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess as fit_lowess

from consts import HINGE_COLOR, LOWESS_COLOR, REFERENCE_COLOR, SCATTER_COLOR, STEP_COLOR
from utils.statistics.find_threshold import ThresholdResult

# each threshold fit also gets its own line style, so the lines stay distinguishable where they overlap
THRESHOLD_STYLES = {
	'step': {'color': STEP_COLOR, 'ls': ':'},
	'hinge': {'color': HINGE_COLOR, 'ls': '--'},
}


def scatterplot(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	lowess: bool = True,
	thresholds: Sequence[ThresholdResult] = (),
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot a scatter of target vs feature with a fitted linear line, an optional LOWESS curve, and optional threshold fits.

	Args:
		df: DataFrame holding both columns. Rows missing either value are dropped.
		feature: Column plotted on the x axis.
		target: Column plotted on the y axis.
		lowess: Draw a LOWESS curve, which follows the local average without assuming a shape.
		thresholds: Step or hinge fits (from find_threshold) to draw, each with a faint vertical guide line at its threshold.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	data = df[[feature, target]].dropna()
	x, y = data[feature], data[target]

	if ax is None:
		_, ax = plt.subplots()

	ax.scatter(x, y, color=SCATTER_COLOR, alpha=0.35, s=12, linewidths=0)

	# 300 evenly spaced x values used to draw the fits, plus each threshold so a step jumps and a hinge bends right at it
	xs = np.sort(np.append(np.linspace(x.min(), x.max(), 300), [result.threshold for result in thresholds]))

	# slope is in target units per feature unit; r is unitless in [-1, 1]
	fit = stats.linregress(x, y)
	ax.plot(
		xs,
		fit.intercept + fit.slope * xs,
		color=REFERENCE_COLOR,
		lw=2,
		label=f'Linear: slope = {fit.slope:.2f}, r = {fit.rvalue:.2f}, R² = {fit.rvalue**2:.2f}',
	)

	if lowess:
		curve = fit_lowess(y, x)  # rows of (x, smoothed y), sorted by x
		ax.plot(curve[:, 0], curve[:, 1], color=LOWESS_COLOR, lw=2, label='LOWESS')

	for result in thresholds:
		style = THRESHOLD_STYLES[result.kind]
		ax.plot(
			xs,
			result.predict(x=xs),
			**style,
			lw=2,
			label=f'{result.kind.capitalize()} @ {result.threshold:.2f}: R² = {result.r_squared:.2f}',
		)
		ax.axvline(result.threshold, **style, lw=1, alpha=0.5)  # faint guide at the threshold

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

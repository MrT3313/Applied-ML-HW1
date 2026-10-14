from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consts import R2_GAIN_MARGIN
from utils.statistics.shape_fits import FAMILIES, reference_r2, shape_fits
from utils.plotting.histogram import histogram
from utils.plotting.scatterplot import scatterplot
from utils.plotting.shape_r2_bars import shape_r2_bars
from utils.plotting.qq import qq
from utils.plotting.box_plot import box_plot, box_plot_vs_target
from utils.plotting.count_plot import count_plot
from utils.plotting.scatter_matrix import scatter_matrix
from utils.plotting.correlation_matrix import correlation_matrix
# from utils.plotting.loss_curve import loss_curve
# from utils.plotting.predicted_vs_actual import predicted_vs_actual

def continuous_single_feature_plots(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	bins: int = 30,
	shapes: bool = True,
	lowess: bool = True,
):
	"""Plot one feature's histogram with a normal overlay, its normal Q-Q plot, its box plot, its scatter against the
	target, and the R² of every candidate shape of it.

	Args:
		df: DataFrame holding both columns.
		feature: Column to plot.
		target: Column the feature is plotted against in the scatter plot and the candidate shapes are fitted to.
		bins: Number of histogram bins.
		shapes: Fit every candidate shape (from shape_fits), plot each one's R² gain over the straight line as a bar,
			and draw on the scatter plot the best shape of each family when it beats the straight line's R² by at least
			R2_GAIN_MARGIN. When False, the bars are left out and the scatter plot shows the straight line and LOWESS only.
		lowess: Draw a LOWESS curve on the scatter plot.

	Returns:
		The Figure, its Axes keyed by chart name ([histogram, qq], [box_plot, scatterplot], [shape_r2]), and the
		candidate shapes (None unless shapes is True).
	"""
	mosaic = [
		['histogram', 'qq'],
		['box_plot', 'scatterplot'],
	]
	height_ratios = [2, 2]
	fits, drawn = None, []

	if shapes:
		mosaic.append(['shape_r2', 'shape_r2'])  # one chart spanning both columns
		height_ratios.append(2)
		fits = shape_fits(df=df, feature=feature, target=target)
		references = reference_r2(df=df, feature=feature, target=target)

		# the best shape of each family, kept only when it beats the straight line by at least the margin
		for family in FAMILIES:
			best = max((shape for shape in fits if shape.family == family), key=lambda shape: shape.r_squared)
			if best.r_squared - references['linear'] >= R2_GAIN_MARGIN:
				drawn.append(best)

	fig, axes = plt.subplot_mosaic(
		mosaic,
		figsize=(14, 6 * sum(height_ratios) / 2),
		height_ratios=height_ratios,
	)

	histogram(series=df[feature], bins=bins, ax=axes['histogram'])
	qq(series=df[feature], ax=axes['qq'])
	box_plot(series=df[feature], ax=axes['box_plot'])
	scatterplot(df=df, feature=feature, target=target, lowess=lowess, fits=drawn, ax=axes['scatterplot'])
	if shapes:
		shape_r2_bars(feature=feature, fits=fits, references=references, ax=axes['shape_r2'])

	fig.tight_layout()

	return fig, axes, fits

def categorical_single_feature_plots(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	by: Optional[str] = None,
):
	"""Plot one categorical feature's row count per level above a box plot of the target for each level.

	Args:
		df: DataFrame holding the columns.
		feature: Categorical column to plot.
		target: Column whose distribution is plotted for each level.
		by: Second categorical column that splits every level in both charts, one color per level of it.
			When omitted, the levels are not split.

	Returns:
		The Figure and its Axes keyed by chart name ([count_plot], [box_plot_vs_target]).
	"""
	fig, axes = plt.subplot_mosaic(
		[
			['count_plot'],
			['box_plot_vs_target'],  # stacked at full width so the boxes of a 12 level feature split by a second column fit
		],
		figsize=(14, 10),
	)

	count_plot(df=df, feature=feature, by=by, ax=axes['count_plot'])
	box_plot_vs_target(df=df, feature=feature, target=target, levels=True, by=by, ax=axes['box_plot_vs_target'])

	fig.tight_layout()

	return fig, axes

def continuous_distribution_plots(
	*,
	series: pd.Series,
	bins: int = 30,
):
	"""Plot one column's histogram with a normal overlay beside its normal Q-Q plot.

	Use it for the target, which has no other column to be plotted against.

	Args:
		series: Numeric values to plot.
		bins: Number of histogram bins.

	Returns:
		The Figure and its array of 2 Axes ([histogram, Q-Q]).
	"""
	fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

	histogram(series=series, bins=bins, ax=axes[0])
	qq(series=series, ax=axes[1])

	fig.tight_layout()

	return fig, axes
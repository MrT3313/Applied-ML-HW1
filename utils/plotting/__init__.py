from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from consts import R2_GAIN_MARGIN
from utils.statistics.shape_fits import category_fits, category_reference_r2, reference_r2, shape_fits
from utils.plotting.histogram import histogram
from utils.plotting.scatterplot import scatterplot
from utils.plotting.shape_r2_bars import shape_r2_bars
from utils.plotting.qq import qq
from utils.plotting.box_plot import box_plot, box_plot_vs_target
from utils.plotting.count_plot import count_plot
from utils.plotting.scatter_matrix import scatter_matrix
from utils.plotting.correlation_matrix import correlation_matrix
from utils.plotting.parameter_search import parameter_search
from utils.plotting.parameter_cross import parameter_cross
from utils.plotting.loss_curve import loss_curve, loss_curve_comparison

def continuous_single_feature_plots(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
):
	"""Plot one feature's histogram with a normal overlay, its normal Q-Q plot, its box plot, its scatter against the
	target, and the R² of every candidate shape of it.

	Every candidate shape (from shape_fits) gets its R² gain over the straight line as a bar, and the scatter plot
	shows the straight line, LOWESS and the best shape of each family when it beats the straight line's R² by at least
	R2_GAIN_MARGIN.

	Args:
		df: DataFrame holding both columns.
		feature: Column to plot.
		target: Column the feature is plotted against in the scatter plot and the candidate shapes are fitted to.

	Returns:
		The Figure, its Axes keyed by chart name ([histogram, qq], [box_plot, scatterplot], [shape_r2]), and the
		candidate shapes.
	"""
	mosaic = [
		['histogram', 'qq'],
		['box_plot', 'scatterplot'],
		['shape_r2', 'shape_r2'],  # one chart spanning both columns
	]
	fits = shape_fits(df=df, feature=feature, target=target)
	references = reference_r2(df=df, feature=feature, target=target)

	# the best shape of each family, kept only when it beats the straight line by at least the margin
	drawn = []
	for family in dict.fromkeys(shape.family for shape in fits):  # the families in the fits, in menu order
		best = max((shape for shape in fits if shape.family == family), key=lambda shape: shape.r_squared)
		if best.r_squared - references['linear'] >= R2_GAIN_MARGIN:
			drawn.append(best)

	fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 18))

	histogram(series=df[feature], ax=axes['histogram'])
	qq(series=df[feature], ax=axes['qq'])
	box_plot(series=df[feature], ax=axes['box_plot'])
	scatterplot(df=df, feature=feature, target=target, fits=drawn, ax=axes['scatterplot'])
	shape_r2_bars(feature=feature, fits=fits, references=references, ax=axes['shape_r2'])

	fig.tight_layout()

	return fig, axes, fits

def categorical_single_feature_plots(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	by: Optional[str] = None,
	period: Optional[float] = None,
):
	"""Plot one categorical feature's row count per level above a box plot of the target for each level, and the
	adjusted R² of every candidate encoding of it.

	Args:
		df: DataFrame holding the columns.
		feature: Categorical column to plot, stored as numbers.
		target: Column whose distribution is plotted for each level and the candidate encodings are fitted to.
		by: Second categorical column that splits every level in both charts, one color per level of it, and is added
			to every candidate encoding as a control. When omitted, the levels are not split.
		period: Number of levels in one full cycle, ex 12 for Month, which adds the cyclic candidates. When omitted,
			they are left out.

	Every candidate encoding (from category_fits) gets its adjusted R² gain over the straight line as a bar, and the
	box plot marks the best encoding of each family when it beats the straight line's adjusted R² by at least
	R2_GAIN_MARGIN. Both are skipped for a feature with 2 levels: every encoding of it is the same 0/1 column, so there
	is nothing to compare, and only the first two charts are drawn.

	Returns:
		The Figure and its Axes keyed by chart name ([count_plot], [box_plot_vs_target], [shape_r2]).
	"""
	mosaic = [
		['count_plot'],
		['box_plot_vs_target'],  # stacked at full width so the boxes of a 12 level feature split by a second column fit
	]
	drawn = []

	encodings = df[feature].nunique() > 2  # with 2 levels every encoding is the same 0/1 column
	if encodings:
		mosaic.append(['shape_r2'])
		fits = category_fits(df=df, feature=feature, target=target, period=period, control=by)
		references = category_reference_r2(df=df, feature=feature, target=target, control=by)

		# the best encoding of each family, kept only when it beats the straight line by at least the margin
		for family in dict.fromkeys(shape.family for shape in fits):  # the families in the fits, in menu order
			best = max((shape for shape in fits if shape.family == family), key=lambda shape: shape.adjusted_r_squared)
			if best.adjusted_r_squared - references['linear'] >= R2_GAIN_MARGIN:
				drawn.append(best)

	fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 5 * len(mosaic)))

	count_plot(df=df, feature=feature, by=by, ax=axes['count_plot'])
	box_plot_vs_target(df=df, feature=feature, target=target, by=by, fits=drawn, ax=axes['box_plot_vs_target'])
	if encodings:
		shape_r2_bars(
			feature=feature,
			fits=fits,
			references=references,
			adjusted=True,
			title=f'{feature} Candidate Encodings: Adjusted R² Gain over the Straight Line' + (f' (with {by})' if by else ''),
			ax=axes['shape_r2'],
		)

	fig.tight_layout()

	return fig, axes

def continuous_distribution_plots(
	*,
	series: pd.Series,
):
	"""Plot one column's histogram with a normal overlay beside its normal Q-Q plot.

	Use it for the target, which has no other column to be plotted against.

	Args:
		series: Numeric values to plot.

	Returns:
		The Figure and its array of 2 Axes ([histogram, Q-Q]).
	"""
	fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

	histogram(series=series, ax=axes[0])
	qq(series=series, ax=axes[1])

	fig.tight_layout()

	return fig, axes
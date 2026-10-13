from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.statistics.find_threshold import (
	THRESHOLD_KINDS,
	ThresholdKind,
	ThresholdResult,
	find_threshold,
)
from utils.plotting.histogram import histogram
from utils.plotting.scatterplot import scatterplot
from utils.plotting.qq import qq
from utils.plotting.box_plot import box_plot, box_plot_vs_target
# from utils.plotting.scatter_matrix import scatter_matrix
# from utils.plotting.correlation_matrix import correlation_matrix
# from utils.plotting.loss_curve import loss_curve
# from utils.plotting.predicted_vs_actual import predicted_vs_actual

def single_feature_plots(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	bins: int = 30,
	threshold: Optional[Literal["auto"]] = None,
	min_group_frac: float = 0.05,
	lowess: bool = True,
):
	"""Plot one feature's histogram with a normal overlay, its normal Q-Q plot, its box plot, and its scatter against the target.

	Args:
		df: DataFrame holding both columns.
		feature: Column to plot.
		target: Column the feature is plotted against in the correlation chart and, when a threshold is given, split into groups in the box plot.
		bins: Number of histogram bins.
		threshold: "auto" searches for the best step and hinge thresholds separately, splits the target box plot at
			each one, and draws both fits on the scatter plot. When omitted, the target box plot shows all rows only
			and no threshold fits are drawn.
		min_group_frac: Smallest share of rows allowed on each side of a split when threshold is "auto".
		lowess: Draw a LOWESS curve on the correlation chart.

	Returns:
		The Figure, its Axes keyed by chart name ([histogram, qq], [box_plot], [box_plot_vs_target, scatterplot]), and the step and hinge
		fits keyed by kind (None unless threshold is "auto").
	"""
	results = None
	box_thresholds = {}

	if threshold == "auto":
		results = {
			kind: find_threshold(df=df, feature=feature, target=target, kind=kind, min_group_frac=min_group_frac)
			for kind in THRESHOLD_KINDS
		}
		box_thresholds = {kind: result.threshold for kind, result in results.items()}

	fig, axes = plt.subplot_mosaic(
		[
			['histogram', 'qq'],
			['box_plot', 'box_plot'],  # one chart spanning both columns
			['box_plot_vs_target', 'scatterplot'],
		],
		figsize=(14, 14),
		height_ratios=[2, 1, 2],
	)

	histogram(series=df[feature], bins=bins, ax=axes['histogram'])
	qq(series=df[feature], ax=axes['qq'])
	box_plot(series=df[feature], ax=axes['box_plot'])
	box_plot_vs_target(df=df, feature=feature, target=target, thresholds=box_thresholds, ax=axes['box_plot_vs_target'])
	scatterplot(
		df=df, feature=feature, target=target,
		lowess=lowess,
		thresholds=list(results.values()) if results else (),
		ax=axes['scatterplot'],
	)

	fig.tight_layout()

	return fig, axes, results

def distribution_plots(
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
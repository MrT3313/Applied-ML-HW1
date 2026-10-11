from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from OLD.utils.plotting.histogram import histogram
from OLD.utils.plotting.scatterplot import scatterplot
from OLD.utils.plotting.qq import qq
from OLD.utils.plotting.box_plot import box_plot
from OLD.utils.plotting.scatter_matrix import scatter_matrix
from OLD.utils.plotting.correlation_matrix import correlation_matrix
from OLD.utils.statistics.find_threshold import (
	THRESHOLD_KINDS,
	ThresholdKind,
	ThresholdResult,
	find_threshold,
	fit_threshold,
)

def single_feature_plots(
	*,
	df: pd.DataFrame,
	feature: str,
	target: str,
	bins: int = 30,
	threshold: Optional[float] | Literal["auto"] = None,
	min_group_frac: float = 0.05,
	lowess: bool = True,
) -> tuple[plt.Figure, np.ndarray, Optional[dict[ThresholdKind, ThresholdResult]]]:
	"""Plot one feature's histogram with a normal overlay, its normal Q-Q plot, its box plot, and its scatter against the target.

	Args:
		df: DataFrame holding both columns.
		feature: Column to plot.
		target: Column the feature is plotted against in the correlation chart and, when a threshold is given, split into groups in the box plot.
		bins: Number of histogram bins.
		threshold: Feature value that splits the box plot into two target groups, with step and hinge fits at that value
			drawn on the correlation chart. "auto" searches for the best step and hinge thresholds separately, and the
			box plot uses the step one. When omitted, the box plot shows the feature alone and no threshold fits are drawn.
		min_group_frac: Smallest share of rows allowed on each side of a split when threshold is "auto".
		lowess: Draw a LOWESS curve on the correlation chart.

	Returns:
		The Figure, its 2x2 array of Axes ([histogram, box plot], [Q-Q, correlation]), and the step and hinge
		fits keyed by kind (None when threshold is omitted).
	"""
	results = None
	box_threshold = None

	if threshold == "auto":
		results = {
			kind: find_threshold(df=df, feature=feature, target=target, kind=kind, min_group_frac=min_group_frac)
			for kind in THRESHOLD_KINDS
		}
		# the box plot compares the mean of each side, which is what the step fit optimizes
		box_threshold = results["step"].threshold
	elif threshold is not None:
		results = {
			kind: fit_threshold(df=df, feature=feature, target=target, kind=kind, threshold=threshold)
			for kind in THRESHOLD_KINDS
		}
		box_threshold = threshold

	fig, axes = plt.subplots(2, 2, figsize=(14, 11))

	histogram(series=df[feature], bins=bins, ax=axes[0, 0])
	qq(series=df[feature], ax=axes[0, 1])
	box_plot(df=df, feature=feature, target=target, threshold=box_threshold, ax=axes[1, 0])
	scatterplot(
		df=df, feature=feature, target=target,
		lowess=lowess,
		thresholds=list(results.values()) if results else (),
		ax=axes[1, 1],
	)

	fig.tight_layout()

	return fig, axes, results

def distribution_plots(
	*,
	series: pd.Series,
	bins: int = 30,
) -> tuple[plt.Figure, np.ndarray]:
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

__all__ = [
	'single_feature_plots',
	'distribution_plots',

	'histogram',
	'scatterplot',
	'qq',
	'box_plot',
	'scatter_matrix',
	'correlation_matrix',
]

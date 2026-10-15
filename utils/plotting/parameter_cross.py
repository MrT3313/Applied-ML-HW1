from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consts import BASELINE_COLOR


def parameter_cross(
	*,
	runs: dict,
	lines: str,
	x: str,
	log_x: bool = False,
	reference: Optional[tuple[str, pd.DataFrame]] = None,
	ymax: Optional[float] = None,
) -> plt.Axes:
	"""Plot the mean validation RMSE of every pair of values tried for two parameters: one line per value of the first
	parameter, across the values of the second.

	Only the means are drawn: a thin line per fold, as in parameter_search, would be one line per fold and per value of
	the first parameter. The star marks the pair with the lowest mean.

	Args:
		runs: Results from PIPELINES.cv_rmse keyed by (value of lines, value of x), in the order tried. The values of x
			must be numbers.
		lines: Name of the parameter drawn as one line per value.
		x: Name of the parameter on the x axis.
		log_x: Put the x values on a log scale, for values spread over several powers of 10.
		reference: (name, result) of a model to draw as a dotted line at its mean validation RMSE, ex the baseline.
		ymax: Highest RMSE shown. A run that diverges can reach RMSEs that flatten every other line against the axis:
			anything above ymax is drawn at ymax as a triangle. None shows everything.

	Returns:
		The Axes the chart was drawn on.
	"""
	_, ax = plt.subplots(figsize=(9, 4.5))

	mean = {pair: run['validation rmse'].mean() for pair, run in runs.items()}
	shown = {pair: value if ymax is None else min(value, ymax) for pair, value in mean.items()}

	for line in dict.fromkeys(pair[0] for pair in runs):  # the values of lines, in the order tried
		pairs = [pair for pair in runs if pair[0] == line]
		xs = np.array([pair[1] for pair in pairs], dtype=float)
		(drawn,) = ax.plot(xs, [shown[pair] for pair in pairs], '-o', linewidth=2, markersize=4, label=f'{lines}={line}')
		above = [pair for pair in pairs if ymax is not None and mean[pair] > ymax]
		if above:
			ax.plot([pair[1] for pair in above], [ymax] * len(above), '^', color=drawn.get_color(), markersize=8)

	best = min(mean, key=mean.get)
	ax.plot(best[1], shown[best], '*', color='black', markersize=14, label=f'lowest mean: {lines}={best[0]}, {x}={best[1]:g}')
	if ymax is not None and any(value > ymax for value in mean.values()):
		ax.plot([], [], '^', color='black', label=f'above {ymax:g} (off the chart)')
	if reference is not None:
		name, run = reference
		ax.axhline(run['validation rmse'].mean(), color=BASELINE_COLOR, linestyle=':', linewidth=1.5, label=f'{name}, validation mean')

	if log_x:
		ax.set_xscale('log')
	ax.set_xlabel(x + (' (log scale)' if log_x else ''))
	ax.set_ylabel('Mean validation RMSE')
	ax.set_title(f'{lines} × {x}: mean validation RMSE')
	ax.grid(alpha=0.3)
	ax.legend(fontsize='small')

	return ax

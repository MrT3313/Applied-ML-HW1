import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from consts import VERDICT_COLORS
from utils.pipelines.cv_rmse import Folds
from utils.statistics.fold_difference import fold_difference


def fold_deltas(
	*,
	runs: dict[str, pd.DataFrame],
	baseline: str,
	names: list[str],
	folds: Folds,
) -> plt.Axes:
	"""Plot, for each of several candidates, its change in validation RMSE against the baseline on every fold.

	One row per candidate, in the order given: a small dot per fold, and a diamond at the mean change (Δrmse) with a
	±1 SE bar. A bar that crosses 0 is within noise. Colors give the verdict (see STATISTICS.fold_difference).

	Args:
		runs: Results from PIPELINES.cv_rmse keyed by name, holding the baseline and every candidate.
		baseline: Name of the result every candidate is compared with.
		names: The candidates, top to bottom.
		folds: The folds every result was scored on, as row positions.

	Returns:
		The Axes the chart was drawn on.
	"""
	_, ax = plt.subplots(figsize=(14, 1.5 + 0.45 * len(names)))  # taller with more rows

	rows = np.arange(len(names))
	for row, name in zip(rows, names):
		difference = fold_difference(baseline=runs[baseline], candidate=runs[name], folds=folds)
		color = VERDICT_COLORS[difference.verdict]
		ax.plot(difference.diff, np.full(len(difference.diff), row), 'o', color=color, alpha=0.4, markersize=4)
		ax.errorbar(difference.mean, row, xerr=difference.se, fmt='D', color=color, markersize=7, capsize=4, linewidth=2)

	ax.axvline(0, color='black', linewidth=1)
	ax.set_yticks(rows, labels=names)
	ax.invert_yaxis()  # first name on top
	ax.set_xlabel(f'validation RMSE change vs {baseline} (below 0 = better)')
	ax.set_title(f'Change against {baseline} on every fold: dots are folds, diamonds are Δrmse ±1 SE')
	ax.grid(alpha=0.3, axis='x')
	ax.legend(
		handles=[Patch(color=color, label=verdict) for verdict, color in VERDICT_COLORS.items()],
		loc='upper center',
		bbox_to_anchor=(0.5, 0),
		# a fixed 40 points below the axes, so it clears the x axis label at any number of rows
		bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, y=-40, units='points'),
		ncol=len(VERDICT_COLORS),
	)

	return ax

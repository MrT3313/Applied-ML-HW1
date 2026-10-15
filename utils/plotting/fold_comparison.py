from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consts import BASELINE_COLOR, CANDIDATE_COLOR, VERDICT_COLORS
from utils.pipelines.cv_rmse import Folds
from utils.statistics.fold_difference import fold_difference


def fold_comparison(
	*,
	runs: dict[str, pd.DataFrame],
	baseline: str,
	candidate: str,
	folds: Folds,
	fold_labels: Optional[list[str]] = None,
	repeat_size: Optional[int] = None,
	decimals: int = 4,
) -> tuple[plt.Figure, np.ndarray]:
	"""Plot every fold's RMSE for a baseline and a candidate, beside the candidate's change on each fold.

	Left: the validation (dots) and training (x's) RMSE of both models on each fold. A training RMSE far below the
	validation RMSE means the model fits its training rows better than new rows.
	Right: the candidate's validation RMSE minus the baseline's on each fold, green where the candidate is better and
	red where it is worse, with the mean change (dashed line) and the ±1 SE noise band (gray). A dashed line inside
	the band is noise, and the title gives the verdict (see STATISTICS.fold_difference).

	Args:
		runs: Results from PIPELINES.cv_rmse keyed by name, holding the baseline and the candidate.
		baseline: Name of the model compared against.
		candidate: Name of the model compared.
		folds: The folds both were scored on, as row positions.
		fold_labels: A label for each fold on the x axis, ex its first validation date. When omitted, folds are
			numbered from 1.
		repeat_size: Folds per repeat when the folds were cut several times (Q1: 4), to draw a line between repeats.
		decimals: Decimals of the numbers in the legends.

	Returns:
		The Figure and its 2 Axes ([RMSE per fold, change per fold]).
	"""
	fold = np.arange(1, len(folds) + 1)
	fig, (ax_scores, ax_diff) = plt.subplots(1, 2, figsize=(18, 4.5))

	for name, color in ((baseline, BASELINE_COLOR), (candidate, CANDIDATE_COLOR)):
		run = runs[name]
		ax_scores.plot(fold, run['validation rmse'], 'o', color=color, label=f'{name} validation (mean {run["validation rmse"].mean():.{decimals}f})')
		ax_scores.plot(fold, run['train rmse'], 'x', color=color, label=f'{name} train (mean {run["train rmse"].mean():.{decimals}f})')
	ax_scores.set_ylabel('RMSE')
	ax_scores.set_title('RMSE per fold')

	difference = fold_difference(baseline=runs[baseline], candidate=runs[candidate], folds=folds)
	ax_diff.bar(fold, difference.diff, color=[VERDICT_COLORS['candidate' if d < 0 else 'worse'] for d in difference.diff])
	ax_diff.axhline(0, color='black', linewidth=1)
	ax_diff.axhspan(-difference.se, difference.se, color=VERDICT_COLORS['within noise'], alpha=0.25, label=f'noise band: ±1 SE (±{difference.se:.{decimals}f})')
	ax_diff.axhline(difference.mean, color='black', linestyle='--', linewidth=1.5, label=f'Δrmse ({difference.mean:+.{decimals}f})')
	ax_diff.set_ylabel(f'validation RMSE change vs {baseline}\n(below 0 = better)')
	ax_diff.set_title(f'{candidate}: better on {difference.better} of {len(fold)} folds, {difference.verdict}')

	for ax in (ax_scores, ax_diff):
		if fold_labels is None:
			ax.set_xticks(fold)
		else:
			ax.set_xticks(fold, labels=fold_labels, rotation=30, ha='right')
		if repeat_size:
			for boundary in np.arange(repeat_size, len(fold), repeat_size) + 0.5:
				ax.axvline(boundary, color='gray', linewidth=0.5)
			ax.set_xlabel(f'Fold ({repeat_size} per repeat, {len(fold) // repeat_size} repeats)')
		else:
			ax.set_xlabel('Fold')
		ax.grid(alpha=0.3, axis='y')
		# below the chart, one column per model, and lower still under slanted fold labels
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2 if fold_labels is None else -0.35), ncol=2)

	fig.tight_layout()
	return fig, np.array([ax_scores, ax_diff])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from OLD.Q2.utils.cross_validation import test_train_ratio
from OLD.utils.scoring.standard_error import standard_error


def cv_fold_scores(
	*,
	baseline: tuple[np.ndarray, np.ndarray],
	compare: tuple[np.ndarray, np.ndarray],
	folds: list[tuple[pd.Index, pd.Index]],
	dates: pd.Series,
	baseline_name: str = "baseline",
	compare_name: str = "compare",
) -> None:
	"""Plot every time-ordered fold's RMSE for a baseline model and a model compared against it.

	Left: validation and training RMSE of both models on each fold.
	Right: compare's validation RMSE minus baseline's on each fold (below 0 = compare is better on that fold), with
	the mean change (dashed line) and a ±1 SE noise band (gray). A dashed line inside the band is noise.

	The folds run left to right in time order, and each one is labelled with the first date of its validation block.

	Args:
		baseline: (val_scores, train_scores) of the baseline, exactly as cv_model returns them.
		compare: (val_scores, train_scores) of the model compared against the baseline.
		folds: The folds the scores were made with (output of make_time_folds).
		dates: Date of every row, indexed like the data the folds were made from.
		baseline_name: Legend name of the baseline.
		compare_name: Legend name of the compared model.
	"""
	fold = np.arange(1, len(folds) + 1)
	# first validation date of each fold, ex "2012-02-03"
	fold_labels = [f"{dates.loc[val_index].min():%Y-%m-%d}" for _, val_index in folds]
	val_rows = len(folds[0][1])

	fig, (ax_scores, ax_diff) = plt.subplots(1, 2, figsize=(18, 4.5))

	# left: one color per model, dots are validation, x's are training
	for (val_scores, train_scores), name, color in [(baseline, baseline_name, "C0"), (compare, compare_name, "C1")]:
		ax_scores.plot(fold, val_scores, "o", color=color, label=f"{name} val (mean {np.mean(val_scores):.4f})")
		ax_scores.plot(fold, train_scores, "x", color=color, label=f"{name} train (mean {np.mean(train_scores):.4f})")
	ax_scores.set_ylabel("RMSE")
	ax_scores.set_title("RMSE per fold")
	# below the chart (under the x axis label), 2 columns: one per model
	ax_scores.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

	# right: same folds, compare minus baseline. Green = compare better on that fold, red = worse.
	diff = np.asarray(compare[0]) - np.asarray(baseline[0])
	mean_diff = diff.mean()

	# noise level: corrected SE of the mean difference, with the folds' own validation / training row ratio
	se = standard_error(diff=diff, test_train_ratio=test_train_ratio(folds=folds))

	# one-SE rule: the mean has to clear 1 SE to count, otherwise keep the simpler model
	if mean_diff < -se:
		verdict = f"really better than {baseline_name}"
	elif mean_diff > se:
		verdict = f"really worse than {baseline_name}"
	else:
		verdict = "noise: keep the simpler model"

	ax_diff.bar(fold, diff, color=np.where(diff < 0, "C2", "C3"))
	ax_diff.axhline(0, color="black", linewidth=1)
	ax_diff.axhspan(-se, se, color="gray", alpha=0.25, label=f"noise band: ±1 SE (±{se:.4f})")
	ax_diff.axhline(mean_diff, color="black", linestyle="--", linewidth=1.5, label=f"mean change ({mean_diff:+.4f})")
	ax_diff.set_ylabel(f"val RMSE change vs {baseline_name}\n(below 0 = better)")
	ax_diff.set_title(f"{compare_name}: better on {int(np.sum(diff < 0))} of {len(diff)} folds | {verdict}")
	ax_diff.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

	for ax in (ax_scores, ax_diff):
		ax.set_xticks(fold)
		ax.set_xticklabels(fold_labels)
		ax.set_xlabel(f"Fold, by first validation date ({len(folds)} time-ordered folds, {val_rows} validation days each)")
		ax.grid(alpha=0.3, axis="y")

	plt.tight_layout()
	plt.show()

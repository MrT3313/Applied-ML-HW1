import numpy as np
import pandas as pd

from utils.scoring.scorers import GREATER_IS_BETTER


def results_df(
	results: dict[str, dict[str, np.ndarray]],
	*,
	baseline: str,
	decimals: int = 3,
) -> pd.DataFrame:
	"""Summarize per-fold scores into one row per intervention, with a noise threshold against the baseline.

	Every intervention is compared with the baseline through the paired per-fold rmse differences
	d = rmse(intervention) - rmse(baseline), which needs both scored on the same folds in the same order. Every
	intervention's validation rows must match the baseline's, fold by fold.

	Columns:
		"{metric} mean": mean of each metric over the folds.
		"train rmse mean": mean rmse on the training rows of each fold, when results holds "train_rmse" (from
			evaluate_pipeline with return_train_score). Close to "rmse mean" means little overfitting.
		"Δrmse vs {baseline}": mean of d. Negative means lower error than the baseline.
		"Δrmse SE": standard error of that mean with the Nadeau and Bengio correction,
			sqrt(var(d) * (1/J + n_test/n_train)), where J is the fold count and n_test/n_train is the mean ratio of
			validation to training rows over the folds. The two pipelines can train on different rows (ex one drops
			incomplete rows, the other imputes them), so the larger of their two ratios is used, the more cautious
			SE. The folds share most of their training rows, so their scores are correlated and the plain var(d)/J
			would understate the noise.
		"verdict": "baseline" for the baseline row. Otherwise "candidate" when Δrmse is below -1 SE, "worse" when
			Δrmse is above +1 SE, else "within noise". These follow the one-standard-error rule: an intervention
			whose Δrmse is not at least one SE below 0 does not earn its extra complexity, so the baseline is kept.
			"worse" marks the ones that are clearly worse than the baseline, not just tied with it.

	Args:
		results: Per-fold scores from evaluate_pipeline, keyed by intervention name. Each must hold
			"validation_rows", "n_train" and "n_test". Keys that are not metrics (ex "estimator") are ignored.
		baseline: Name of the intervention the rmse difference is measured against. It must be in results.
		decimals: Number of decimals the table is rounded to.

	Returns:
		One row per intervention, sorted by mean rmse and rounded to decimals. The baseline row has Δrmse 0 and NaN for
		SE.
	"""
	base = results[baseline]
	base_rmse = np.asarray(base["rmse"])
	base_ratio = np.mean(np.asarray(base["n_test"]) / np.asarray(base["n_train"]))
	for name, scores in results.items():
		same_rows = len(scores["validation_rows"]) == len(base["validation_rows"]) and all(
			rows.equals(base_rows) for rows, base_rows in zip(scores["validation_rows"], base["validation_rows"])
		)
		assert same_rows, f"{name} was not validated on the same rows as {baseline}, so their scores cannot be paired"
	delta_col = f"Δrmse vs {baseline}"

	rows = {}
	for name, scores in results.items():
		row = {}
		for metric in GREATER_IS_BETTER:
			if metric in scores:
				row[f"{metric} mean"] = np.mean(scores[metric])
			if metric == "rmse" and "train_rmse" in scores:
				row["train rmse mean"] = np.mean(scores["train_rmse"])

		if name == baseline:
			row.update({delta_col: 0.0, "Δrmse SE": np.nan})
		else:
			d = np.asarray(scores["rmse"]) - base_rmse
			assert len(d) == len(base_rmse), f"{name} and {baseline} differ in fold count"
			n_folds = len(d)
			# the larger ratio of the two pipelines gives the more cautious SE
			ratio = max(base_ratio, np.mean(np.asarray(scores["n_test"]) / np.asarray(scores["n_train"])))
			se = np.sqrt(np.var(d, ddof=1) * (1 / n_folds + ratio))
			row.update({delta_col: np.mean(d), "Δrmse SE": se})
		rows[name] = row

	df = pd.DataFrame.from_dict(rows, orient="index")
	df.index.name = "intervention"
	others = df.index != baseline

	df["verdict"] = "within noise"
	df.loc[others & (df[delta_col] < -df["Δrmse SE"]), "verdict"] = "candidate"
	df.loc[others & (df[delta_col] > df["Δrmse SE"]), "verdict"] = "worse"
	df.loc[baseline, "verdict"] = "baseline"

	metric_cols = [c for c in df.columns if c.endswith(" mean")]
	df = df[metric_cols + [delta_col, "Δrmse SE", "verdict"]]
	return df.sort_values("rmse mean").round(decimals)

"""The noise rule: is a run's rmse different from the baseline's by more than fold to fold noise?

Pure math on score arrays. entrypoints/compare.py fetches the scores from W&B and stores the result there.
"""
import numpy as np
import pandas as pd

from src.utils.standard_error import standard_error


def compare(fold_rmse: dict[str, np.ndarray], *, baseline: str, n_splits: int) -> pd.DataFrame:
    """Compare every run with a baseline run through their paired per-fold rmse differences.

    Each run must have been scored on the same folds as the baseline, in the same order.

    Columns:
        "rmse_mean": mean rmse over the folds.
        "delta": mean of the per-fold differences, rmse(run) - rmse(baseline). Negative means lower error.
        "se": corrected standard error of that mean (Nadeau and Bengio), from standard_error.
        "delta_over_se": delta divided by se.
        "verdict": "baseline" for the baseline row. "identical" when every fold matches the baseline.
            Otherwise the one-standard-error rule: "candidate" when delta is below -1 se, "worse" when it
            is above +1 se, else "within noise", where the simpler of the two models is kept.

    Args:
        fold_rmse: Per-fold rmse of each run, keyed by run name.
        baseline: Name of the run the others are measured against. It must be in fold_rmse.
        n_splits: Folds per seed the scores were made with.

    Returns:
        One row per run, sorted by rmse_mean.
    """
    base = np.asarray(fold_rmse[baseline])

    rows = {}
    for name, scores in fold_rmse.items():
        scores = np.asarray(scores)
        assert len(scores) == len(base), f"{name} and {baseline} differ in fold count"
        diff = scores - base
        row = {"rmse_mean": np.mean(scores), "delta": np.mean(diff), "se": np.nan, "delta_over_se": np.nan}

        if name == baseline:
            row["verdict"] = "baseline"
        elif np.allclose(diff, 0, atol=1e-9):
            # the same predictions up to float rounding, so there is no noise to measure
            row["delta"] = 0.0
            row["verdict"] = "identical"
        else:
            row["se"] = standard_error(diff=diff, n_splits=n_splits)
            row["delta_over_se"] = row["delta"] / row["se"]
            if row["delta"] < -row["se"]:
                row["verdict"] = "candidate"
            elif row["delta"] > row["se"]:
                row["verdict"] = "worse"
            else:
                row["verdict"] = "within noise"
        rows[name] = row

    # rounded so that runs tied up to float rounding keep the order they were given in
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.sort_values("rmse_mean", key=lambda column: column.round(9), kind="stable")

"""Compare every run of a sweep with the sweep's baseline run and write the verdict onto each run.

W&B shows runs side by side but has no statistical test. This fills that gap: it reads each run's per-fold
rmse back from the server, applies the noise rule in src/compare.py, and saves the result into each run's
summary. The comparison then appears as columns in the sweep's own runs table:
    verdict         baseline | identical | within noise | candidate | worse
    delta_rmse      mean per-fold rmse difference from the baseline. Negative is better.
    delta_se        corrected standard error of that difference
    delta_over_se   delta_rmse / delta_se. Below -1 is a real improvement, above +1 is really worse.
    baseline_run    name of the run it was measured against

Usage, from inside Q1_WandB. The run-sweep target calls this automatically after the agent finishes:
    make -f Makefile.WeightsAndBiases compare SWEEP_ID=<sweep_id> [BASELINE=<run_id>]
"""
import sys

import pandas as pd
import wandb

from src.compare import compare
from src.grid import COMBO_KEY

PROJECT = "happiness"
# two runs are only scored on the same folds when all of these match
FOLD_KEYS = ("data_train", "cv_seeds", "cv_n_splits")


def main(sweep: str, baseline_id: str | None = None):
    """Compare every finished run of a sweep with its baseline and write the verdict onto each run.

    Args:
        sweep: A sweep ID, or its full entity/project/sweep_id path.
        baseline_id: ID of the run to measure against. Defaults to the first run of the sweep.
    """
    # wandb.Api() works on runs that already finished, unlike wandb.init which starts a new one
    api = wandb.Api()
    sweep = api.sweep(sweep if "/" in sweep else f"{api.default_entity}/{PROJECT}/{sweep}")
    runs = sorted((run for run in sweep.runs if run.state == "finished"), key=lambda run: run.created_at)
    if not runs:
        sys.exit(f"sweep {sweep.id} has no finished runs to compare, see the run errors above")
    by_id = {run.id: run for run in runs}
    # a grid starts with the first value of every parameter, so by default that run is the baseline
    baseline = by_id[baseline_id] if baseline_id else runs[0]

    for run in runs:
        for key in FOLD_KEYS:
            assert run.config[key] == baseline.config[key], f"{run.id} differs from the baseline in {key}"

    # history is what a run logged step by step, here the 20 per-fold scores from entrypoints/run.py
    fold_rmse = {}
    for run in runs:
        history = run.history(keys=["fold", "fold/rmse"]).sort_values("fold")
        fold_rmse[run.id] = history["fold/rmse"].to_numpy()

    stats = compare(fold_rmse, baseline=baseline.id, n_splits=baseline.config["cv_n_splits"])

    # write the comparison onto each run, so it shows as columns in the sweep's own runs table
    for run in runs:
        row = stats.loc[run.id]
        values = {
            "baseline_run": baseline.name,
            "verdict": row["verdict"],
            "delta_rmse": row["delta"],
            "delta_se": row["se"],
            "delta_over_se": row["delta_over_se"],
        }
        # the baseline and identical runs have no se, and a missing value is left out instead of saved
        for key, value in values.items():
            if pd.notna(value):
                run.summary[key] = value if isinstance(value, str) else float(value)
        run.summary.update()      # sends the changed summary to the server

    # print only the config keys that differ between the runs
    keys = sorted(key for key in baseline.config if not key.startswith("_") and key != COMBO_KEY)
    varied = [key for key in keys if len({str(run.config[key]) for run in runs}) > 1]
    configs = pd.DataFrame({run.id: {key: run.config[key] for key in varied} for run in runs}).T
    table = configs.join(stats).loc[stats.index]
    table.index.name = "run"
    print(table.round(4).to_string())
    print(f"\nbaseline: {baseline.name} ({baseline.id})")
    print(f"decide here: {sweep.url}")


if __name__ == "__main__":
    main(*sys.argv[1:])

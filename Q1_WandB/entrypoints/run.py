"""Train and score one configuration as one W&B run. Launched by a sweep agent, once per combination.

The agent starts this file with `python -m entrypoints.run` and no arguments. The combination to try
arrives through wandb.init as one config key, "combo": the JSON of every setting this run should use,
built by entrypoints/sweep.py. It is unpacked here into ordinary config keys.

What the run records, and where each part shows up in the UI:
    config    the inputs: data version, folds, the sweep's choices. Table columns and filters.
    history   one point per fold: fold/rmse, fold/mae, fold/r2. Line charts.
    summary   one number per run: rmse_mean, rmse_sd, mae_mean, r2_mean. Table columns and sweep panels.
    lineage   use_artifact marks the dataset version as an input of this run.

Not meant to be run by hand: without a sweep there is no combo to unpack.
"""
import json
from pathlib import Path

import wandb

from src.config import load_config, validate
from src.data import load_data
from src.grid import COMBO_KEY
from src.train import train
from src.utils.make_folds import make_folds

ROOT = Path(__file__).resolve().parent.parent


def main():
    """Open a run, train on the config the sweep handed out, and record the scores."""
    defaults = load_config(ROOT / "configs" / "base.yaml")["run"]
    # inside a sweep W&B uses the sweep's project and prints a warning that this project= is ignored
    with wandb.init(project="happiness", job_type="train", config=defaults) as run:
        # unpack the sweep's combo into ordinary config keys, so the UI can filter and sort by each one
        combo = json.loads(run.config[COMBO_KEY])
        run.config.update(combo)
        config = validate({**defaults, **combo})
        # name the run after what its sweep chose: the model first, then the other keys alphabetically,
        # ex "model_kind=dummy | data_missing=drop | dummy_strategy=median | scaler=none". A key that does
        # not apply to this run (None) is left out
        chosen = sorted((key for key in combo if combo[key] is not None), key=lambda key: (key != "model_kind", key))
        run.name = " | ".join(f"{key}={combo[key]}" for key in chosen)
        # download the pinned dataset version, which also records it as this run's input
        train_dir = run.use_artifact(config["data_train"]).download()
        X, y = load_data(train_dir)
        # built from every row, before any are dropped, so all configs are validated on the same rows
        folds = make_folds(X, seeds=config["cv_seeds"], n_splits=config["cv_n_splits"])

        # the only line that does machine learning. Nothing below src/ knows about W&B
        result = train(config, X, y, folds)

        # history: one point per fold, charted against the fold number instead of the step counter
        run.define_metric("fold", hidden=True)
        run.define_metric("fold/*", step_metric="fold")
        for row in result.fold_metrics:
            run.log(row)
        # summary: one number per run (rmse_mean, rmse_sd, ...), shown as table columns and never charted
        run.summary.update(result.metrics)


if __name__ == "__main__":
    main()

"""Turn one chosen training run into a saved model artifact. Run after reading the sweep's comparison.

Sweep runs only score a config with cross validation and keep no model. This refits the chosen run's exact
config on every usable training row and stores the fitted pipeline on the server as the artifact
"happiness-regressor", labelled with the alias "candidate".

An alias is a movable label on one artifact version. Other code asks for "happiness-regressor:candidate"
without knowing the version number, and promoting a different run later moves the label to the new version.

Usage, from inside Q1_WandB. The run ID is the last part of a run's URL:
    make -f Makefile.WeightsAndBiases promote RUN=<run_id>
"""
import sys
from pathlib import Path

import joblib
import wandb

from src.config import validate
from src.data import load_data
from src.grid import COMBO_KEY
from src.train import fit

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "happiness"
MODEL_NAME = "happiness-regressor"


def main(run_id: str):
    """Refit the config of a finished training run on all training rows and log the model.

    Args:
        run_id: ID of the training run whose config should become the model.
    """
    # read the chosen run's config back from the server, so the model cannot drift from what was scored
    api = wandb.Api()
    winner = api.run(f"{api.default_entity}/{PROJECT}/{run_id}")
    # keys starting with "_" are W&B's own, and combo is only how the sweep delivered the settings
    winner_config = {
        key: value for key, value in winner.config.items() if not key.startswith("_") and key != COMBO_KEY
    }

    with wandb.init(project=PROJECT, job_type="promote", config=winner_config) as run:
        config = validate(dict(run.config))
        train_dir = run.use_artifact(config["data_train"]).download()
        X, y = load_data(train_dir)

        model = fit(config, X, y)

        # a local copy to upload from. MODELS/ is git ignored, the artifact on the server is the record
        model_path = ROOT / "MODELS" / "model.joblib"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)

        # metadata travels with the model version and points back at the evidence for choosing it
        artifact = wandb.Artifact(
            name=MODEL_NAME,
            type="model",
            metadata={"source_run": run_id, "cv_rmse_mean": winner.summary.get("rmse_mean")},
        )
        artifact.add_file(str(model_path))
        run.log_artifact(artifact, aliases=["candidate"])


if __name__ == "__main__":
    main(sys.argv[1])

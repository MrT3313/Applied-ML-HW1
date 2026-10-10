"""Score the promoted model on the test set. The only script that reads the test data.

Run it once, for the one model already chosen through cross validation. It reports a number and must not be
used to pick between models: choosing by test score turns the test set into a validation set, and its score
stops being an honest estimate of how the model does on unseen data.

Usage, from inside Q1_WandB:
    make -f Makefile.WeightsAndBiases final-eval [MODEL=happiness-regressor:candidate]
"""
import sys
from pathlib import Path

import joblib
import wandb

from src.data import complete_rows, load_data
from src.evaluate import score

PROJECT = "happiness"
# pinned to a version so the result always names its exact test data. Update it after registering again
DATA_TEST = "happiness-test:v0"


def main(model_ref: str = "happiness-regressor:candidate"):
    """Download a model artifact and the test set, score the model, and log the test metrics.

    Args:
        model_ref: Model artifact as name:alias or name:version.
    """
    # the config records which model and which test data produced the numbers
    with wandb.init(project=PROJECT, job_type="test", config={"model": model_ref, "data_test": DATA_TEST}) as run:
        # both downloads are recorded as inputs, so lineage shows this is the run that read the test set
        model_dir = run.use_artifact(model_ref).download()
        test_dir = run.use_artifact(DATA_TEST).download()

        model = joblib.load(Path(model_dir) / "model.joblib")
        X_all, y_all = load_data(test_dir)
        # the model cannot predict a row with a missing value, so those rows are counted and left out
        X, y = complete_rows(X_all, y_all)

        metrics = {f"test_{metric}": value for metric, value in score(model, X, y).items()}
        run.log({**metrics, "test_rows": len(X), "test_rows_skipped": len(X_all) - len(X)})


if __name__ == "__main__":
    main(*sys.argv[1:])

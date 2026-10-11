"""Upload the train and test CSVs to W&B as versioned dataset artifacts. Run once, before any sweep.

An artifact is a named, versioned set of files stored on the W&B server. Each upload of new content becomes
the next version (v0, v1, ...), and uploading unchanged content creates no new version. Runs ask for a
dataset by name and version (ex "happiness-train:v0"), which ties every result to the exact data behind it.

Usage, from inside Q1_WandB:
    make -f Makefile.WeightsAndBiases register-data-artifacts

Deleting this script's run in the UI can also delete the dataset versions it created. After registering
again, check the version number in the Artifacts tab and make data_train in configs/base.yaml and DATA_TEST
in entrypoints/final_eval.py match it.
"""
from pathlib import Path

import wandb

DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"

# artifact name on the server -> local file to upload
DATASETS = {
    "happiness-train": DATA_DIR / "happiness_data_train.csv",
    "happiness-test": DATA_DIR / "happiness_data_test.csv",
}

# a run is one tracked execution. job_type labels what kind of step it was, here a data upload
with wandb.init(project="happiness", job_type="data") as run:
    for name, path in DATASETS.items():
        # type groups artifacts in the UI, so datasets stay apart from the "model" artifacts
        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_file(str(path))
        # logging it uploads the file and records this run as its producer in the lineage graph
        run.log_artifact(artifact)

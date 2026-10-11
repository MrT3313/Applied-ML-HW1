"""Read a dataset CSV and select its rows. The same code serves the train and the test data.

It only sees a local folder. Downloading that folder from a W&B artifact is the entrypoint's job.
"""
from pathlib import Path

import pandas as pd

# the column being predicted. Every other column is a feature
TARGET = "Life Ladder"


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the one CSV in a downloaded dataset artifact folder and split it into features and target.

    The rows keep the index read_csv gives them (0, 1, 2, ...), which is what the folds refer to.

    Args:
        data_dir: Folder holding exactly one CSV, ex the folder an artifact download returns.

    Returns:
        The feature columns X and the target column y, with every row of the file.
    """
    (csv_path,) = Path(data_dir).glob("*.csv")
    df = pd.read_csv(csv_path)
    return df.drop(columns=[TARGET]), df[TARGET]


def complete_rows(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Keep only the rows with no missing feature value. The kept rows keep their index labels."""
    complete = X.notna().all(axis=1)
    return X[complete], y[complete]

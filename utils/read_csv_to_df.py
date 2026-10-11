from pathlib import Path
from typing import Union

import pandas as pd


def read_csv_to_df(*, path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file.
        **kwargs: Extra keyword arguments passed through to pandas.read_csv.

    Returns:
        A DataFrame containing the CSV contents.

    Raises:
        FileNotFoundError: If no file exists at the given path.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)

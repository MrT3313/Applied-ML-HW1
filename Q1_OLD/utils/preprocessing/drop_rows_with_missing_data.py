import pandas as pd

def drop_rows_with_missing_data(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    columns: list | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    subset = X if columns is None else X[columns]
    mask = subset.notna().all(axis=1)
    return X[mask], y[mask]
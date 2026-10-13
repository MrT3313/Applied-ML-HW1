import pandas as pd
from sklearn.model_selection import train_test_split


def chronological_split(*, df: pd.DataFrame, date_col: str, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Split rows into train (earliest dates) and test (latest dates), with no shuffling."""
	ordered = df.sort_values(date_col)
	train_df, test_df = train_test_split(ordered, test_size=test_size, shuffle=False)  # shuffle=False keeps the date order

	return train_df, test_df
import pandas as pd
from sklearn.model_selection import train_test_split


def chronological_split(*, df: pd.DataFrame, date_col: str, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Split rows into train (earliest dates) and test (latest dates), with no shuffling.

	Args:
		df: DataFrame to split.
		date_col: Column the rows are ordered by before splitting.
		test_size: Share of rows held out as test, taken from the end of the date range.

	Returns:
		(train_df, test_df). When every date is distinct, every train date is earlier than every test date.
	"""
	ordered = df.sort_values(date_col)
	train_df, test_df = train_test_split(ordered, test_size=test_size, shuffle=False)  # shuffle=False keeps the date order

	return train_df, test_df

import numpy as np
import pandas as pd


def add_shape_columns(X: pd.DataFrame) -> pd.DataFrame:
	"""Turn the 11 raw attribute columns into the 12 columns Part D selected (BEST_FEATURES).

	Used as the first step of Part D's final pipeline (through FunctionTransformer), so the saved model takes the
	raw attributes and builds its own shape columns. The threshold and the levels are fixed values chosen in
	Part D. Nothing is learned from the rows passed in, so a level that is absent from X still gets its column.

	Args:
		X: DataFrame holding the 11 raw attribute columns.

	Returns:
		A new DataFrame with exactly the BEST_FEATURES columns, in order: 8 raw attributes, the FeelsLike flat top
		(raw FeelsLike removed) and the Season one-hot columns for levels 2 to 4 (the Season code removed).
		IsWorkday is dropped.
	"""
	X = X.copy()
	# flat top: max(0, t - x)
	X["FeelsLike flat top 0.57"] = np.maximum(0, 0.57 - X["FeelsLike"])
	# one-hot: 1 when the code equals the level, else 0. The first level gets no column.
	for level in (2, 3, 4):
		X[f"Season {level}"] = (X["Season"] == level).astype(float)
	return X[[
		"Temp", "Humidity", "Wind", "Year", "IsHoliday", "Month", "Weekday", "Weather",
		"FeelsLike flat top 0.57",
		"Season 2", "Season 3", "Season 4",
	]]

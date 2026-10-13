import numpy as np
import pandas as pd


def error_by_group(actual: pd.Series, predictions: pd.DataFrame, *, by: pd.Series) -> pd.DataFrame:
	"""Rows, mean error and rmse of each model within each group of rows.

	Args:
		actual: Actual target values.
		predictions: One column of predicted values per model, with the same index as actual.
		by: Group of each row (ex a month, or a band from pd.cut), with the same index as actual.

	Returns:
		One row per group, in group order. Columns: ("rows", ""), then ("mean error", model) and ("rmse", model)
		for each model. The error is predicted minus actual, so a positive mean error is a model that predicts too
		high on that group. A group with no rows has 0 rows and NaN scores.
	"""
	assert predictions.index.equals(actual.index) and by.index.equals(actual.index), "all three must have the same index"

	error = predictions.sub(actual, axis=0)
	grouped = error.groupby(by, observed=False)
	table = pd.concat(
		{"mean error": grouped.mean(), "rmse": grouped.agg(lambda errors: np.sqrt((errors**2).mean()))},
		axis=1,
	)
	table.insert(0, ("rows", ""), grouped.size())
	return table

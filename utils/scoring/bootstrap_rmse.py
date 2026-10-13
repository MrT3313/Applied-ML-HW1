import numpy as np
import pandas as pd


def bootstrap_rmse(
	actual: pd.Series,
	predictions: pd.DataFrame,
	*,
	n_resamples: int = 10_000,
	block_size: int = 1,
	seed: int = 42,
) -> pd.DataFrame:
	"""rmse of each model on rows resampled with replacement, to show how much a score depends on which rows it has.

	Every resample has as many rows as actual, and every model is scored on the same resampled rows, so the
	difference between two columns of the result is a paired difference. The models are not refit: this measures
	the noise from the rows that are scored, not from the rows a model was trained on.

	Args:
		actual: Actual target values.
		predictions: One column of predicted values per model, with the same index as actual.
		n_resamples: Number of resamples.
		block_size: Rows drawn together. 1 draws single rows, which treats the rows as independent. Above 1, each
			draw is a run of block_size consecutive rows (a moving block), for rows in time order whose errors
			carry over from one row to the next.
		seed: Seed of the random generator that draws the rows.

	Returns:
		One row per resample and one column per model: that model's rmse on that resample.
	"""
	assert predictions.index.equals(actual.index), "predictions and actual must have the same index"

	squared_error = (predictions.to_numpy() - actual.to_numpy()[:, None]) ** 2
	n_rows = len(squared_error)
	n_blocks = -(-n_rows // block_size)  # blocks needed to cover n_rows, rounded up

	rng = np.random.default_rng(seed)
	starts = rng.integers(0, n_rows - block_size + 1, size=(n_resamples, n_blocks))
	# each start becomes block_size consecutive positions, and a resample is cut back to n_rows
	rows = (starts[:, :, None] + np.arange(block_size)).reshape(n_resamples, -1)[:, :n_rows]
	return pd.DataFrame(np.sqrt(squared_error[rows].mean(axis=1)), columns=predictions.columns)

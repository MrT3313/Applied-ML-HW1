from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, r2_score

from utils.scoring.scorers import root_mean_squared_error


def prediction_scores(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
	"""Score one set of predictions with rmse, mae and r2.

	Args:
		y_true: Actual target values.
		y_pred: Predicted values, in the same row order.

	Returns:
		{"rmse", "mae", "r2"}, each in its natural units (errors positive).
	"""
	return {
		"rmse": root_mean_squared_error(y_true, y_pred),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred)),
	}

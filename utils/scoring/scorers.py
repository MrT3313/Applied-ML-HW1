import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import make_scorer, mean_squared_error, r2_score


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
	"""Square root of the mean squared error."""
	return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# errors are better when smaller, so their scorers return the negated metric
GREATER_IS_BETTER: dict[str, bool] = {
	"rmse": False,
	"r2": True,
}

SCORERS = {
	"rmse": make_scorer(root_mean_squared_error, greater_is_better=GREATER_IS_BETTER["rmse"]),
	"r2": make_scorer(r2_score, greater_is_better=GREATER_IS_BETTER["r2"]),
}

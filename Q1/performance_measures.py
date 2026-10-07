from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

Metric = Callable[[ArrayLike, ArrayLike], float]

def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
	return float(np.sqrt(mean_squared_error(y_true, y_pred)))


PERFORMANCE_MEASURES: dict[str, Metric] = {
	"rmse": root_mean_squared_error,
	"mse": mean_squared_error,
	"mae": mean_absolute_error,
}

SCORERS = {
	name: make_scorer(metric, greater_is_better=False)
	for name, metric in PERFORMANCE_MEASURES.items()
}
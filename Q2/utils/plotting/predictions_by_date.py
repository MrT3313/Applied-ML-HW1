import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = ["#2a78d6", "#d1661a"]  # blue, orange


def predictions_by_date(*, dates: pd.Series, actual: pd.Series, predictions: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
	"""Plot actual and predicted values by date, with each model's error (predicted minus actual) below.

	Args:
		dates: Date of each row.
		actual: Actual target values, with the same index as dates. Its name labels the axes.
		predictions: One column of predicted values per model (at most 2), with the same index as dates.

	Returns:
		The Figure and its array of 2 Axes ([values, errors]).
	"""
	assert predictions.shape[1] <= len(MODEL_COLORS), "one color per model is needed"

	fig, (top, bottom) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

	top.plot(dates, actual, color="black", linewidth=1.5, label="actual")
	# the same color for a model in both panels
	for color, name in zip(MODEL_COLORS, predictions.columns):
		top.plot(dates, predictions[name], color=color, linewidth=1.5, label=name)
		bottom.plot(dates, predictions[name] - actual, color=color, linewidth=1.5, label=name)
	bottom.axhline(0, color="black", linewidth=1, label="predicted = actual")

	top.set_title(f"Actual and predicted {actual.name} by date")
	top.set_ylabel(actual.name)
	bottom.set_ylabel(f"Predicted minus actual {actual.name}")
	bottom.set_xlabel("Date")
	for ax in (top, bottom):
		ax.grid(alpha=0.3)
		ax.legend(loc="upper left", ncol=predictions.shape[1] + 1)

	fig.tight_layout()
	return fig, np.array([top, bottom])

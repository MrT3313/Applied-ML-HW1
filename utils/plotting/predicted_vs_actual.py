import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCATTER_COLOR = "#2a78d6"  # blue


def predicted_vs_actual(*, actual: pd.Series, predictions: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
	"""Plot predicted against actual values, one panel per model on shared axes.

	A dot on the diagonal is a perfect prediction. A dot above it was predicted too high, and a dot below it too low.

	Args:
		actual: Actual target values. Its name labels the axes.
		predictions: One column of predicted values per model, with the same index as actual.

	Returns:
		The Figure and its array of Axes (one per model).
	"""
	# one range for both axes of every panel, so the diagonal means the same thing in each
	low = min(actual.min(), predictions.min().min())
	high = max(actual.max(), predictions.max().max())
	margin = 0.04 * (high - low)
	low, high = low - margin, high + margin

	n_models = predictions.shape[1]
	fig, axes = plt.subplots(1, n_models, figsize=(9 * n_models, 8), sharex=True, sharey=True, squeeze=False)
	axes = axes.ravel()
	for ax, name in zip(axes, predictions.columns):
		ax.scatter(actual, predictions[name], s=14, alpha=0.6, color=SCATTER_COLOR, linewidths=0)
		ax.plot([low, high], [low, high], color="black", linewidth=1, label="predicted = actual")
		ax.set_xlim(low, high)
		ax.set_ylim(low, high)
		ax.set_aspect("equal")
		ax.set_xlabel(f"Actual {actual.name}")
		ax.set_ylabel(f"Predicted {actual.name}")
		ax.set_title(f"{name}: predicted vs actual")
		ax.grid(alpha=0.3)
		ax.legend(loc="upper left")

	fig.tight_layout()
	return fig, axes

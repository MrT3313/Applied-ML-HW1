from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LINE_STYLES: dict[str, dict] = {
	"train rmse": {"color": "#2a78d6", "marker": "x", "label": "train"},  # blue
	"validation rmse": {"color": "#d1661a", "marker": "o", "label": "validation"},  # orange
}

# a panel showing more epochs than this draws its lines without a marker per epoch
MAX_MARKED_EPOCHS = 100


def loss_curve(
	*,
	curve: Union[pd.DataFrame, dict[str, pd.DataFrame]],
	zoom_from: Optional[int] = None,
	log_x: bool = False,
	title: str = "RMSE by epoch",
) -> tuple[plt.Figure, np.ndarray]:
	"""Plot training and validation rmse against the epoch, as the mean over the folds with a band of ±1 sd.

	Args:
		curve: Per fold, per epoch scores from PIPELINES.loss_curve (index levels "fold" and "epoch", columns
			"train rmse" and "validation rmse"). Or several of them keyed by name (ex one per degree), drawn one
			panel each on a shared y axis so they can be compared.
		zoom_from: Add a second panel that starts at this epoch, for when the first epochs set a scale that
			hides the later changes. None draws one panel. With several curves no panel is added: every panel
			starts at this epoch.
		log_x: Put the epochs on a log scale, for curves of thousands of epochs where most of the change happens
			in the first ones.
		title: Title of the first panel. With several curves each panel is titled with its curve's name.

	Returns:
		The Figure and its array of Axes (one, two with zoom_from, or one per curve).
	"""
	several = isinstance(curve, dict)
	if several:
		# one panel per curve: (curve, first epoch shown, title)
		panels = [
			(one, zoom_from, name if zoom_from is None else f"{name} (epoch {zoom_from} on)")
			for name, one in curve.items()
		]
	else:
		panels = [(curve, None, title)]
		if zoom_from is not None:
			panels.append((curve, zoom_from, f"{title} (epoch {zoom_from} on)"))

	fig, axes = plt.subplots(1, len(panels), figsize=(min(9 * len(panels), 18), 4.5), squeeze=False, sharey=several)
	axes = axes.ravel()
	for ax, (one, first_epoch, panel_title) in zip(axes, panels):
		by_epoch = one.groupby(level="epoch")
		mean, sd = by_epoch.mean(), by_epoch.std(ddof=1)
		n_folds = one.index.get_level_values("fold").nunique()
		if first_epoch is None:
			first_epoch = mean.index[0]

		shown = mean.loc[first_epoch:]
		marked = len(shown) <= MAX_MARKED_EPOCHS
		for column, style in LINE_STYLES.items():
			ax.plot(
				shown.index, shown[column],
				color=style["color"], marker=style["marker"] if marked else None, markersize=4, linewidth=2, label=style["label"],
			)
			band = sd.loc[first_epoch:, column]
			ax.fill_between(shown.index, shown[column] - band, shown[column] + band, color=style["color"], alpha=0.12)
		if log_x:
			ax.set_xscale("log")
		ax.set_xlabel("Epoch (log scale)" if log_x else "Epoch")
		# a shared y axis is labeled once, on the first panel
		if not several or ax is axes[0]:
			ax.set_ylabel(f"RMSE (mean of {n_folds} folds, band ±1 sd)")
		ax.set_title(panel_title)
		ax.grid(alpha=0.3)
		ax.legend()

	fig.tight_layout()
	return fig, axes

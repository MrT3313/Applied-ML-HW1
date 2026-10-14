from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from consts import FAMILY_COLORS, LOWESS_COLOR, REFERENCE_COLOR
from utils.statistics.shape_fits import ShapeFit


def shape_r2_bars(
	*,
	feature: str,
	fits: Sequence[ShapeFit],
	references: dict[str, float],
	adjusted: bool = False,
	title: Optional[str] = None,
	ax: Optional[plt.Axes] = None,
) -> plt.Axes:
	"""Plot every candidate shape's R² gain over the straight line as a horizontal bar, highest R² on top.

	A bar to the right of 0 beats the straight line, and the dashed line marks how far the ceiling (LOWESS for a
	continuous feature, one-hot for a categorical one) gets.

	Args:
		feature: Column the shapes were built from, used in the default title.
		fits: Candidate shapes (from shape_fits or category_fits).
		references: R² keyed by 'linear', then the ceiling's name (from reference_r2 or category_reference_r2). The
			dashed line is named after the second key.
		adjusted: Use adjusted R² (labeled "adj. R²") for the bars, their text and the legend, as category_fits is
			graded. When omitted, plain R² is used.
		title: Title of the chart. When omitted, it names the feature's candidate shapes.
		ax: Axes to draw on. A new figure and axes are created when omitted.

	Returns:
		The Axes the chart was drawn on.
	"""
	linear = references['linear']
	ceiling_name, ceiling = list(references.items())[1]  # the second key names the dashed line
	metric = 'adj. R²' if adjusted else 'R²'

	def score(shape: ShapeFit) -> float:
		return shape.adjusted_r_squared if adjusted else shape.r_squared

	ordered = sorted(fits, key=score, reverse=True)
	gains = np.array([score(shape) - linear for shape in ordered])
	rows = np.arange(len(ordered))

	if ax is None:
		_, ax = plt.subplots()

	ax.barh(rows, gains, height=0.7, color=[FAMILY_COLORS[shape.family] for shape in ordered])
	ax.set_yticks(rows, labels=[shape.label for shape in ordered])
	ax.invert_yaxis()  # put the highest R² on top

	# Each bar's R², its gain (the bar's length) and column count, written just right of the bar (or of 0 for a bar that
	# goes left), so the text never runs into the shape names. 4 decimals, because the axis is zoomed in on the gains and
	# shows differences that 2 decimals would round away.
	for row, shape, gain in zip(rows, ordered, gains):
		ax.annotate(
			f'{metric} = {score(shape):.4f} ({gain:+.4f}), {shape.columns} column{"" if shape.columns == 1 else "s"}',
			xy=(max(gain, 0), row),
			xytext=(4, 0),  # 4 points right of the bar end
			textcoords='offset points',
			va='center',
			fontsize='small',
		)

	ax.axvline(0, color=REFERENCE_COLOR, lw=1.5)
	ax.axvline(ceiling - linear, color=LOWESS_COLOR, lw=1.5, ls='--')

	# leave room right of the longest bar for its text
	left = min(gains.min(), 0)
	right = max(gains.max(), ceiling - linear, 0)
	ax.set_xlim(left - 0.02 * (right - left), right + 0.35 * (right - left))
	ax.grid(axis='x', alpha=0.3)
	ax.set_axisbelow(True)  # draw the grid behind the bars

	ax.set_title(title or f'{feature} Candidate Shapes: R² Gain over the Straight Line')
	ax.set_xlabel(f'{metric} gain over the straight line')

	# Legend centered below the axes, outside the plot area: the reference lines, then one entry per family drawn
	families = {shape.family for shape in fits}
	handles = [
		Line2D([], [], color=REFERENCE_COLOR, lw=1.5, label=f'Linear: {metric} = {linear:.4f}'),
		Line2D([], [], color=LOWESS_COLOR, lw=1.5, ls='--', label=f'{ceiling_name}: {metric} = {ceiling:.4f} ({ceiling - linear:+.4f})'),
		*(Patch(color=color, label=family) for family, color in FAMILY_COLORS.items() if family in families),
	]
	ax.legend(
		handles=handles,
		loc='upper center',  # anchor point on the legend box
		bbox_to_anchor=(0.5, -0.12),  # place that anchor centered, just below the x-axis label
		ncol=len(handles),
		fontsize='small',
		framealpha=0.9,
	)

	return ax

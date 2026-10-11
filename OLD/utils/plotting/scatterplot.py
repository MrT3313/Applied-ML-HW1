from typing import Optional, Sequence

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from OLD.utils.statistics.find_threshold import ThresholdResult

# Each fit gets its own colorblind-safe color, and the threshold fits also get their own line style,
# so lines stay distinguishable where they overlap.
LINE_STYLES: dict[str, dict] = {
    "linear": {"color": "red", "linestyle": "-"},  # red, solid
    "lowess": {"color": "#159a6b", "linestyle": "-"},  # green, solid
    "step": {"color": "#2a78d6", "linestyle": ":"},  # blue, dotted
    "hinge": {"color": "#4a3aa7", "linestyle": "--"},  # violet, dashed
}
LINE_WIDTH = 2.5

# white outline under each fit line so it separates from the scatter and from other lines it crosses
LINE_OUTLINE = [pe.Stroke(linewidth=LINE_WIDTH + 2.5, foreground="white"), pe.Normal()]

SCATTER_COLOR = "#9e9e9e"  # neutral gray so the points recede behind the colored fit lines


def _plot_threshold(ax: plt.Axes, result: ThresholdResult, x_min: float, x_max: float) -> None:
    """Draw a threshold fit as its fitted segments on each side of t plus a thin vertical guide line at t."""
    style = LINE_STYLES[result.kind]
    t = result.threshold

    if result.kind == "step":
        low_mean, high_mean = result.coefficients
        details = f"means {low_mean:.2f} / {high_mean:.2f}"
        # one line that is flat on each side and jumps at t, so the legend shows the fit's own style
        x_line = np.array([x_min, t, t, x_max])
        y_line = np.array([low_mean, low_mean, high_mean, high_mean])
    else:
        _, b, c = result.coefficients
        details = f"slopes {b:.2f} / {b + c:.2f}"
        x_line = np.array([x_min, t, x_max])
        y_line = result.predict(x_line)

    ax.plot(
        x_line, y_line, **style,
        linewidth=LINE_WIDTH, path_effects=LINE_OUTLINE, zorder=4,
        label=f"{result.kind.capitalize()} @ {t:.2f}: {details}, R² = {result.r_squared:.2f}",
    )

    # thin, faint guide at t; unlabeled because the fit line above carries the legend entry
    ax.axvline(t, color=style["color"], linestyle=style["linestyle"], linewidth=1, alpha=0.5, zorder=3)


def scatterplot(
    *,
    df: pd.DataFrame,
    feature: str,
    target: str,
    lowess: bool = True,
    thresholds: Sequence[ThresholdResult] = (),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a scatter of target vs feature with a fitted linear line, an optional LOWESS curve, and optional threshold fits.

    Args:
        df: DataFrame holding both columns. Rows missing either value are dropped.
        feature: Column plotted on the x axis.
        target: Column plotted on the y axis.
        lowess: Draw a LOWESS curve, which follows the local average without assuming a shape.
        thresholds: Step or hinge fits (from find_threshold or fit_threshold) to draw, each as its fitted
            segments plus a faint vertical guide line at its threshold.
        ax: Axes to draw on. A new figure and axes are created when omitted.

    Returns:
        The Axes the chart was drawn on.
    """
    data = df[[feature, target]].dropna()

    # slope is in target units per feature unit; r is unitless in [-1, 1]
    fit = stats.linregress(data[feature], data[target])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    sns.regplot(
        data=data, x=feature, y=target,
        color=LINE_STYLES["linear"]["color"],  # also tints the confidence band
        scatter_kws={"color": SCATTER_COLOR, "alpha": 0.35, "s": 12, "linewidths": 0},
        line_kws={
            **LINE_STYLES["linear"],
            "linewidth": LINE_WIDTH,
            "path_effects": LINE_OUTLINE,
            "zorder": 4,
            "label": f"Linear: slope = {fit.slope:.2f}, r = {fit.rvalue:.2f}, R² = {fit.rvalue**2:.2f}",
        },
        ax=ax,
    )

    if lowess:
        sns.regplot(
            data=data, x=feature, y=target,
            lowess=True, scatter=False,
            color=LINE_STYLES["lowess"]["color"],
            line_kws={
                "linestyle": LINE_STYLES["lowess"]["linestyle"],
                "linewidth": LINE_WIDTH,
                "path_effects": LINE_OUTLINE,
                "zorder": 4,
            },
            # with scatter=False seaborn puts regplot's label on the line, overriding line_kws
            label="LOWESS",
            ax=ax,
        )

    for result in thresholds:
        _plot_threshold(ax, result, data[feature].min(), data[feature].max())

    ax.set_title(f"{target} vs {feature} Scatter Plot")

    # Legend centered below the axes, outside the plot area
    ax.legend(
        loc='upper center',  # anchor point on the legend box
        bbox_to_anchor=(0.5, -0.15),  # place that anchor centered, just below the x-axis label
        ncol=2,  # entries are long, so 2 columns keeps the legend short without overflowing the axes width
        fontsize='small',
        framealpha=0.9,
    )

    return ax

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_correlation_to_target(
    *,
    df: pd.DataFrame,
    feature: str,
    target: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a scatter of target vs feature with a fitted regression line, labeled with its slope and Pearson r.

    Args:
        df: DataFrame holding both columns. Rows missing either value are dropped.
        feature: Column plotted on the x axis.
        target: Column plotted on the y axis.
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
        scatter_kws={"alpha": 0.4, "s": 15},
        line_kws={
            "color": "red",
            "label": f"Fit: slope = {fit.slope:.2f}, r = {fit.rvalue:.2f}",
        },
        ax=ax,
    )

    ax.set_title(f"{target} vs {feature}")
    ax.legend(loc='upper left', fontsize='small', framealpha=0.9)

    return ax

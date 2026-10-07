from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_hist_with_normal_dist_overlay(
    *,
    series: pd.Series,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a density histogram with a fitted normal curve, mean line and +/- 1, 2, 3 SD lines.

    Args:
        series: Numeric values to plot. Missing values are dropped.
        bins: Number of histogram bins.
        ax: Axes to draw on. A new figure and axes are created when omitted.

    Returns:
        The Axes the chart was drawn on.
    """
    x = series.dropna()
    mu, sigma = x.mean(), x.std()

    if ax is None:
        _, ax = plt.subplots()

    x.hist(
        bins=bins,
        density=True,  # scale bar heights so the total bar area is 1, putting the histogram on the same scale as the normal curve
        ax=ax,
    )

    # 300 evenly spaced x values used to draw a smooth normal curve
    xs = np.linspace(x.min(), x.max(), 300)

    # Plot Normal Distribution
    ax.plot(
        xs,  # x values
        stats.norm.pdf(xs, mu, sigma),  # y values
        'r-',  # red solid line
        lw=2,  # line width
        label=f'Normal (mean={mu:.2f}, sd={sigma:.2f})',
    )

    # Mean line
    ax.axvline(mu, color='black', lw=2, label='Mean')

    # +/- 1, 2, 3 standard deviations
    styles = {1: '--', 2: '-.', 3: ':'}
    for k in range(1, 4):
        for sign in (-1, 1):
            ax.axvline(mu + sign * k * sigma, color='gray', ls=styles[k], lw=1.5,
                       label=f'±{k} SD' if sign == 1 else None)

    ax.set_title(series.name)
    ax.legend(loc='upper right', fontsize='small', framealpha=0.9)

    return ax

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from consts import DEFAULT_BLUE, MEAN_COLOR, MEDIAN_COLOR, REFERENCE_COLOR


def histogram(
    *,
    series: pd.Series,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a density histogram with a fitted normal curve, mean line, median line and +/- 1, 2, 3 SD lines.

    Args:
        series: Numeric values to plot. Missing values are dropped.
        bins: Number of histogram bins.
        ax: Axes to draw on. A new figure and axes are created when omitted.

    Returns:
        The Axes the chart was drawn on.
    """
    x = series.dropna()
    mu, sigma = x.mean(), x.std()
    median = x.median()

    if ax is None:
        _, ax = plt.subplots()

    x.hist(
        bins=bins,
        density=True,  # scale bar heights so the total bar area is 1, putting the histogram on the same scale as the normal curve
        color=DEFAULT_BLUE,
        ax=ax,
    )

    # 300 evenly spaced x values used to draw a smooth normal curve
    xs = np.linspace(x.min(), x.max(), 300)

    # Plot Normal Distribution
    ax.plot(
        xs,  # x values
        stats.norm.pdf(xs, mu, sigma),  # y values
        color=REFERENCE_COLOR,
        lw=2,  # line width
        label=f'Normal (mean={mu:.2f}, sd={sigma:.2f})',
    )

    # Mean line
    ax.axvline(mu, color=MEAN_COLOR, lw=2, label=f'Mean ({mu:.2f})')

    # Median line
    ax.axvline(median, color=MEDIAN_COLOR, lw=2, label=f'Median ({median:.2f})')

    # +/- 1, 2, 3 standard deviations
    styles = {1: '--', 2: '-.', 3: ':'}
    for k in range(1, 4):
        for sign in (-1, 1):
            ax.axvline(mu + sign * k * sigma, color='gray', ls=styles[k], lw=1.5,
                       label=f'±{k} SD' if sign == 1 else None)

    ax.set_title(f'{series.name} Histogram w/ Normal Distribution Overlay')
    ax.set_xlabel(series.name)
    ax.set_ylabel('Density')

    # Legend centered below the axes, outside the plot area
    ax.legend(
        loc='upper center',  # anchor point on the legend box
        bbox_to_anchor=(0.5, -0.15),  # place that anchor centered, just below the x-axis label
        ncol=3,  # spread entries across columns to keep the legend short
        fontsize='small',
        framealpha=0.9,
    )

    return ax
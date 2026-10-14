HIST_BINS = 30
TEST_SIZE = 0.2
SEEDS = [42, 0, 1, 2, 3]
ALPHAS = [0.001, 0.01, 0.1]
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
R2_GAIN_MARGIN = 0.01  # smallest R² gain over the baseline shape for a shape to be carried forward

# COLORS
DEFAULT_BLUE = '#184f95'
MEDIAN_COLOR = '#33b833'
MEAN_COLOR = 'tab:orange'
REFERENCE_COLOR = 'red'
LOWESS_COLOR = '#159a6b'
STEP_COLOR = '#862d2d'
HINGE_COLOR = '#a655f6'
SCATTER_COLOR = '#9e9e9e'  # neutral gray so the points recede behind the colored fit lines
GROUP_COLORS = [DEFAULT_BLUE, '#5aa9f5']  # one per level of a `by` column, in level order (Year has two levels)
# one per family of candidate shapes (utils/statistics/shape_fits.py), used by both the scatter plot curves and the R² bars
FAMILY_COLORS = {
	're-expression': '#c27c00',
	'piecewise linear': HINGE_COLOR,
	'bins': '#a8473f',
}

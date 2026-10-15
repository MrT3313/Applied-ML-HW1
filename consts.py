TEST_SIZE = 0.2

# D: cross validation and the fixed lists each SGD setting is tried at
SEED = 42  # every fold shuffle and every SGD model
CARRY_MARGIN = 0.01  # the decision rule: an option replaces the baseline (the simpler option) only if its mean validation RMSE is at least 1% lower
CV_SPLITS = 4  # the homework's four-fold cross validation
MAX_ITER = 10_000  # most epochs an SGD model may run; tol stops it earlier
TOL = 1e-3  # scikit-learn's default stopping tolerance, written out
POWER_T = 0.25  # scikit-learn's default for invscaling, written out
LEARNING_RATE_SCHEDULES = ['constant', 'optimal', 'invscaling', 'adaptive']
ETA0S = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # the starting step size
ALPHAS = [1e-3, 1e-1, 10]  # penalty strength, a hundred times apart
L1_RATIO = 0.5  # Elastic Net's share of L1 in the penalty, an even mix

# E: polynomial regression
DEGREES = [1, 2, 3, 4]  # polynomial degrees tried

R2_GAIN_MARGIN = 0.01  # smallest R² gain for a candidate shape to be drawn on the B plots, or a row shown in the C R2 Gain tables
COLLINEARITY_MIN_R = 0.3  # smallest size of r between two attributes for their pair to be shown in the C collinearity table

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
BASELINE_COLOR = DEFAULT_BLUE  # D: the reference line (a model the others are compared with) on the parameter charts
TRAIN_COLOR = DEFAULT_BLUE  # D: training rows of a fold, in the parameter searches and the loss curve
VALIDATION_COLOR = '#d1661a'  # D: validation rows of a fold
LOSS_METHOD_COLORS = {'warm_start': '#a655f6', 'refit': '#159a6b'}  # D: the homework's loop and the exact curve, on the chart that compares them
# one per family of candidate shapes (utils/statistics/shape_fits.py), used by the scatter plot curves, the R² bars and
# the marks on the categorical box plots
FAMILY_COLORS = {
	're-expression': '#c27c00',
	'piecewise linear': HINGE_COLOR,
	'bins': '#a8473f',
	'groups': '#d6336c',
	'cyclic': '#0aa6b8',
}

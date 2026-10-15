TEST_SIZE = 0.2

# D: cross validation and the SGD searches
SEED = 42  # every fold shuffle and every SGD model
CV_SPLITS = 4  # the homework's four-fold cross validation
MAX_ITER = 10_000  # most epochs an SGD model may run; tol stops it earlier
LEARNING_RATE_SCHEDULES = ['constant', 'optimal', 'invscaling', 'adaptive']
ETA0S = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]  # the starting step size
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]  # penalty strength
L1_RATIOS = [0.1, 0.15, 0.3, 0.5, 0.7, 0.9]  # Elastic Net's share of L1 in the penalty

# E: polynomial regression
DEGREES = [1, 2, 3, 4]  # polynomial degrees searched
TOLS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  # stopping tolerance: the drop in training loss (standardized target) an epoch must beat

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
# D: the model a candidate is compared against, the candidate, and the verdict of a per-fold change
BASELINE_COLOR = DEFAULT_BLUE
CANDIDATE_COLOR = '#d1661a'
TRAIN_COLOR = DEFAULT_BLUE  # D: training rows of a fold, in the parameter searches and the loss curve
VALIDATION_COLOR = '#d1661a'  # D: validation rows of a fold
VERDICT_COLORS = {
	'candidate': MEDIAN_COLOR,  # better than the baseline by more than 1 SE
	'within noise': SCATTER_COLOR,
	'worse': REFERENCE_COLOR,  # worse than the baseline by more than 1 SE
}
# one per family of candidate shapes (utils/statistics/shape_fits.py), used by the scatter plot curves, the R² bars and
# the marks on the categorical box plots
FAMILY_COLORS = {
	're-expression': '#c27c00',
	'piecewise linear': HINGE_COLOR,
	'bins': '#a8473f',
	'groups': '#d6336c',
	'cyclic': '#0aa6b8',
}

import numpy as np

from OLD.Q1.consts import COLUMNS


def add_hinge_columns(X):
	"""Turn the 8 raw feature columns into the 11 columns Part D selected (BEST_FEATURES).

	Used as the first step of Part D's final pipeline (through FunctionTransformer), so the saved model takes the
	raw columns and builds its own hinge columns. The thresholds are fixed numbers chosen in Part D. Nothing is
	learned from the rows passed in.

	Args:
		X: DataFrame holding the 8 raw feature columns.

	Returns:
		A new DataFrame: raw Healthy life expectancy removed (its hinge replaces it), 4 hinge columns added.
	"""
	X = X.copy()
	# hinge: max(0, x - t)
	X["GDP hinge 10.3"] = np.maximum(0, X[COLUMNS['LGDP']['col']] - 10.3)
	X["GDP hinge 10.9"] = np.maximum(0, X[COLUMNS['LGDP']['col']] - 10.9)
	X["Life exp hinge 57.5"] = np.maximum(0, X[COLUMNS['HLEAB']['col']] - 57.5)
	X["Social hinge 0.84"] = np.maximum(0, X[COLUMNS['SS']['col']] - 0.84)
	return X.drop(columns=[COLUMNS['HLEAB']['col']])

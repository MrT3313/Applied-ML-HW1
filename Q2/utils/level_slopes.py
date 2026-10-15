import numpy as np


def level_slopes(X: np.ndarray, *, levels: list[float]) -> np.ndarray:
	"""Give a continuous attribute its own slope on each level of a categorical one (an interaction).

	Used as FunctionTransformer(level_slopes, kw_args={'levels': ...}) on two columns: the categorical attribute
	first, then the continuous one. The attribute's own column still gives the slope shared by every level, so
	these columns only add how much each level's slope differs from the first level's.

	Args:
		X: Two columns: the categorical attribute's codes, then the continuous attribute.
		levels: Every level of the categorical attribute, the first being the baseline. Passed in rather than read
			from X, so a fold missing a level still gets the same columns.

	Returns:
		One column per level but the first: the continuous attribute on that level's rows, 0 on the others.
	"""
	X = np.asarray(X, dtype=float)
	return np.column_stack([(X[:, 0] == level) * X[:, 1] for level in levels[1:]])

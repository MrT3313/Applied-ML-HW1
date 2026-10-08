# Smallest |r| for each Pearson correlation strength, strongest first so pcc_strength returns the first one |r| reaches
PCC_THRESHOLDS: dict[str, float] = {'strong': 0.7, 'moderate': 0.4, 'weak': 0.2, 'very weak': 0.0}


def pcc_strength(r: float) -> str:
	"""Return the strength label of correlation coefficient r. The sign of r is ignored.

	Raises:
		StopIteration: r is NaN, which fails every threshold comparison.
	"""
	a = abs(r)
	return next(label for label, t in PCC_THRESHOLDS.items() if a >= t)

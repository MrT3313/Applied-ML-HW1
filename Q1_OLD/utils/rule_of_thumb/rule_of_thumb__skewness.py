from pandas import Series


def rule_of_thumb__skewness(*, series: Series= None, value:float = None):
	assert (series is not None) or (value is not None)

	if series is not None:
		skew = series.skew()
	if value is not None:
		skew = value

	match skew:
		case s if abs(s) < 0.5:
			return "Roughly symmetric"
		case s if 0.5 <= abs(s) <= 1.0:
			return "Moderately skewed"
		case _:
			return "Highly skewed"
def rule_of_thumb__skewness(*, value: float):
	match value:
		case s if abs(s) < 0.5:
			return "Roughly symmetric"
		case s if 0.5 <= abs(s) <= 1.0:
			return "Moderately skewed"
		case _:
			return "Highly skewed"
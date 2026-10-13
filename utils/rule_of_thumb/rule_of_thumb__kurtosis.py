from pandas import Series


def rule_of_thumb__kurtosis(*, series: Series= None, value:float = None):
	"""
		Kurtosis measures tail heaviness (how prone the data is to extreme outliers), not peakedness (Westfall 2014)
		pandas .kurt() reports EXCESS kurtosis (Fisher's definition, kurtosis - 3), so normal tails score 0, not 3
		Excess kurtosis can't go below -2, so negative values never get as large as positive ones

		Thresholds (on excess kurtosis):
			abs(k) <= 1  -> Mesokurtic (Hair et al. 2017)
			1 < k <= 7   -> Leptokurtic
			k > 7        -> Severely leptokurtic (West, Finch & Curran 1995)
			k < -1       -> Platykurtic

		Caution:
			Near 0 means the TAILS look normal, not that the distribution is normal
			(skewed, non-normal distributions can also score 0)
	"""
	assert (series is not None) or (value is not None)

	if series is not None:
		kurt = series.kurt()
	if value is not None:
		kurt = value

	match kurt:
		case k if k < -1.0:
			return "Platykurtic (light tails, fewer outliers)"
		case k if k <= 1.0:
			return "Mesokurtic (near-normal tails)"
		case k if k <= 7.0:
			return "Leptokurtic (heavy tails, more outliers)"
		case _:
			return "Severely leptokurtic (very heavy tails, many outliers)"
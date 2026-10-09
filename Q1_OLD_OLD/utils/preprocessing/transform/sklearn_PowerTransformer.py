from pandas import DataFrame, Series
from sklearn.preprocessing import PowerTransformer

from typing import Literal, Union

def power_transform(
	data: Union[DataFrame, Series],
	method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson"
) -> DataFrame:
	# Learns a lambda per column (1 = none, 0 = log-like, > 1 = fixes left skew). box-cox requires x > 0
	# sklearn transformers need 2D input, so a single column is promoted to a DataFrame
	if isinstance(data, Series):
		data = data.to_frame()

	# standardize=False keeps scaling as its own step
	transformer = PowerTransformer(method=method, standardize=False).set_output(transform="pandas")
	transformed_df = transformer.fit_transform(data)

	return transformed_df

from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler

from typing import Union

def standard_scale(
	data: Union[DataFrame, Series],
	with_mean: bool = True,
	with_std: bool = True
) -> DataFrame:
	# sklearn scalers need 2D input, so a single column is promoted to a DataFrame
	if isinstance(data, Series):
		data = data.to_frame()

	scaler = StandardScaler(with_mean=with_mean, with_std=with_std).set_output(transform="pandas")
	scaled_df = scaler.fit_transform(data)

	return scaled_df

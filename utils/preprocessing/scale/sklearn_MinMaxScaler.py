from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler

from typing import Tuple, Union

def min_max_scaler(
	data: Union[DataFrame, Series],
	feature_range: Tuple[float, float] = (-1, 1)
) -> DataFrame:
	# sklearn scalers need 2D input, so a single column is promoted to a DataFrame
	if isinstance(data, Series):
		data = data.to_frame()

	scaler = MinMaxScaler(feature_range=feature_range).set_output(transform="pandas")
	scaled_df = scaler.fit_transform(data)

	return scaled_df

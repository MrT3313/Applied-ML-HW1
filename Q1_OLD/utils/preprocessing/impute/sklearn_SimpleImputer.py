from pandas import DataFrame, Series
from sklearn.impute import SimpleImputer

from typing import Literal, Union

def simple_impute(
	data: Union[DataFrame, Series],
	strategy: Literal["mean", "median", "most_frequent"]
) -> DataFrame:
	# sklearn imputers need 2D input, so a single column is promoted to a DataFrame
	if isinstance(data, Series):
		data = data.to_frame()

	imputer = SimpleImputer(strategy=strategy).set_output(transform="pandas")
	imputed_df = imputer.fit_transform(data)

	return imputed_df

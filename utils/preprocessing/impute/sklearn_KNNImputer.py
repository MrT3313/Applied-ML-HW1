from pandas import DataFrame
from sklearn.impute import KNNImputer

from typing import Literal

def knn_imputer(
	df: DataFrame,
	n_neighbors: int = 5,
	weights: Literal["uniform", "distance"] = "uniform"
) -> DataFrame:
	# Neighbors are found with euclidean distance across all columns, so pass
	# several related numeric features (ideally on similar scales), not a single column
	imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights).set_output(transform="pandas")
	imputed_df = imputer.fit_transform(df)

	return imputed_df

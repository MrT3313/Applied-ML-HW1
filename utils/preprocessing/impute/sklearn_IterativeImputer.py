from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required before importing IterativeImputer)
from sklearn.impute import IterativeImputer

from typing import Literal

def iterative_imputer(
	df: DataFrame,
	initial_strategy: Literal["mean", "median", "most_frequent"] = "median",
	max_iter: int = 10,
	random_state: int = 0
) -> DataFrame:
	# Each column with missing values is regressed on the other columns, so pass
	# several related numeric features, not a single column
	imputer = IterativeImputer(
		initial_strategy=initial_strategy,
		max_iter=max_iter,
		random_state=random_state
	).set_output(transform="pandas")
	imputed_df = imputer.fit_transform(df)

	return imputed_df

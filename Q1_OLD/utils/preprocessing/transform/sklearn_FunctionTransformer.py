import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import FunctionTransformer

from typing import Union

# sklearn has no log / sqrt / power class, so these wrap numpy in a FunctionTransformer.
# FunctionTransformer learns nothing in fit, so unlike PowerTransformer there is no learned shift:
# log needs x > 0 and sqrt / fractional powers need x >= 0, otherwise numpy returns NaN with only a warning

def raise_to_power(X, exponent: float):
	# np.power is a ufunc and won't take its exponent as a keyword, so kw_args needs a named wrapper
	# 0 < exponent < 1 fixes right skew (smaller is stronger, 0.5 = sqrt), > 1 fixes left skew
	return X ** exponent

def _function_transform(data: Union[DataFrame, Series], transformer: FunctionTransformer) -> DataFrame:
	# sklearn transformers need 2D input, so a single column is promoted to a DataFrame
	if isinstance(data, Series):
		data = data.to_frame()

	# one-to-one keeps the column names, so set_output(transform="pandas") works
	transformer.set_params(feature_names_out="one-to-one").set_output(transform="pandas")
	transformed_df = transformer.fit_transform(data)

	return transformed_df

def log_transform(data: Union[DataFrame, Series]) -> DataFrame:
	return _function_transform(data, FunctionTransformer(np.log))

def sqrt_transform(data: Union[DataFrame, Series]) -> DataFrame:
	return _function_transform(data, FunctionTransformer(np.sqrt))

def exponent_transform(data: Union[DataFrame, Series], exponent: float) -> DataFrame:
	return _function_transform(data, FunctionTransformer(raise_to_power, kw_args={"exponent": exponent}))

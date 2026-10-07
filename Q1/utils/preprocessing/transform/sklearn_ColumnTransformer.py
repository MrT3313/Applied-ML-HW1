from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

def on_columns(transformer: TransformerMixin, columns: list[str]) -> ColumnTransformer:
	# applies transformer to only these columns and passes the rest through unchanged, so a transform can target
	# the features whose skew it fixes (ex: log on the right-skewed ones only)
	# the transformed columns come out first and the passed-through ones after, which a linear model doesn't mind
	return ColumnTransformer([("transform", transformer, columns)], remainder="passthrough")

from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# What build_pipeline does with one attribute: 'drop' leaves it out, 'passthrough' keeps it raw, a transformer
# replaces it with the transformer's columns (ex ShapeFit.transformer), and a (transformer, columns) pair adds the
# transformer's columns built from several attributes (ex an interaction) on top of what each attribute already gives.
Shape = Union[str, BaseEstimator, tuple[BaseEstimator, list[str]]]


def build_pipeline(
	*,
	model: BaseEstimator,
	attributes: list[str],
	shapes: Optional[dict[str, Shape]] = None,
	imputer: Optional[BaseEstimator] = None,
	degree: int = 1,
	expand: Optional[list[str]] = None,
) -> TransformedTargetRegressor:
	"""The unfitted pipeline of one experiment: the imputer (when given), each attribute shaped as in shapes (keyed by
	attribute, or by a name of your choice for a pair; attributes not named stay raw), every column standardized, then
	the model. The target is standardized too and predictions come back in its units, so SGD's step and tol suit any
	target scale. Every step is fit on the training rows only, and fitting works on copies of the steps.

	Above degree 1 the standardized columns are expanded into every product of up to degree of them (PolynomialFeatures,
	no column of 1s since the model fits its own intercept) and standardized again. Standardizing before the expansion
	keeps a column and its square from being near copies, after it gives every expanded column the same scale for SGD's
	step and the penalty. expand limits the expansion to the columns of the attributes it names (the others go to the
	model as they are); with expand, every attribute must be named in shapes.
	"""
	shapes = shapes or {}
	blocks, read_by_pair = [], set()
	for name, shape in shapes.items():
		step, columns = shape if isinstance(shape, tuple) else (shape, [name])
		blocks.append((name, step, [attributes.index(column) for column in columns]))
		if isinstance(shape, tuple):
			read_by_pair.update(columns)

	# The remainder only passes on the columns no block reads, so an attribute read by a pair and not named on its own
	# would be lost: it is passed on raw here.
	for name in sorted(read_by_pair - shapes.keys(), key=attributes.index):
		blocks.append((name, 'passthrough', [attributes.index(name)]))

	columns = [ColumnTransformer(blocks, remainder='passthrough'), StandardScaler()]
	if degree > 1 and expand is None:
		columns += [PolynomialFeatures(degree, include_bias=False), StandardScaler()]
	elif degree > 1:
		every = list(range(len(attributes)))
		expanded = make_pipeline(
			ColumnTransformer([block for block in blocks if block[0] in expand]),
			StandardScaler(),
			PolynomialFeatures(degree, include_bias=False),
		)
		unexpanded = ColumnTransformer([block for block in blocks if block[0] not in expand])
		columns = [ColumnTransformer([('expanded', expanded, every), ('unexpanded', unexpanded, every)]), StandardScaler()]

	steps = [] if imputer is None else [imputer]
	return TransformedTargetRegressor(
		regressor=make_pipeline(*steps, *columns, model),
		transformer=StandardScaler(),
	)

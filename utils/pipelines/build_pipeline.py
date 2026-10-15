from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
) -> TransformedTargetRegressor:
	"""The unfitted pipeline of one experiment: the imputer (when given), each attribute shaped as in shapes (keyed by
	attribute, or by a name of your choice for a pair; attributes not named stay raw), every column standardized, then
	the model. The target is standardized too and predictions come back in its units, so SGD's step and tol suit any
	target scale. Every step is fit on the training rows only, and fitting works on copies of the steps.
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

	steps = [] if imputer is None else [imputer]
	return TransformedTargetRegressor(
		regressor=make_pipeline(*steps, ColumnTransformer(blocks, remainder='passthrough'), StandardScaler(), model),
		transformer=StandardScaler(),
	)

from typing import Optional, Union

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# What build_pipeline does with one attribute: 'drop' leaves it out, 'passthrough' keeps it raw, and a transformer
# replaces it with the transformer's columns (ex ShapeFit.transformer).
Shape = Union[str, BaseEstimator]


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
	attribute; attributes not named stay raw), every attribute's columns standardized, then the model. The target is
	standardized too and predictions come back in its units, so SGD's step and tol suit any target scale. Every step is
	fit on the training rows only, and fitting works on copies of the steps.

	Above degree 1 the standardized columns of the attributes in expand (every attribute when expand is None) are
	expanded into every product of up to degree of them (PolynomialFeatures, no column of 1s since the model fits its
	own intercept) and standardized again. Standardizing before the expansion keeps a column and its square from being
	near copies, after it gives every expanded column the same scale for SGD's step and the penalty. The other
	attributes' columns go to the model standardized, not expanded.

	Any column the imputer adds after the attributes (ex the 0/1 missing flags of MissingIndicator) goes to the model
	as it is, 0 or 1: not standardized, so a rare flag does not become a huge value that throws SGD off, and not
	expanded, since the product of a flag with a column is not a polynomial term. The imputer must keep the attributes
	first and in their order, since the shapes find them by position.
	"""
	shapes = shapes or {}
	every = list(range(len(attributes)))  # the attributes' positions in what the imputer returns

	def shaped(*, names: list[str]) -> ColumnTransformer:
		"""The columns of the attributes in names, each shaped as in shapes; every other column is left out."""
		return ColumnTransformer([(name, shapes.get(name, 'passthrough'), [attributes.index(name)]) for name in names])

	expanded = [name for name in attributes if degree > 1 and (expand is None or name in expand)]
	rest = [name for name in attributes if name not in expanded]

	parts = []
	if expanded:
		parts.append((
			'expanded',
			make_pipeline(shaped(names=expanded), StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler()),
			every,
		))
	if rest:
		parts.append(('not expanded', make_pipeline(shaped(names=rest), StandardScaler()), every))
	columns = ColumnTransformer(parts, remainder='passthrough')  # the remainder: what the imputer adds after the attributes

	steps = [] if imputer is None else [imputer]
	return TransformedTargetRegressor(
		regressor=make_pipeline(*steps, columns, model),
		transformer=StandardScaler(),
	)

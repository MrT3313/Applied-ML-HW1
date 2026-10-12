from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RAW = "raw"
DROP = "drop"


def attribute_pipeline(
	estimator: BaseEstimator,
	*,
	attributes: list[str],
	shapes: dict[str, dict[str, BaseEstimator | list[BaseEstimator]]],
	choice: dict[str, str] | None = None,
	degree: int = 1,
	expand: list[str] | None = None,
	scale: bool = False,
	imputer: BaseEstimator | None = None,
) -> Pipeline:
	"""Build one pipeline from a per attribute feature choice and a model.

	The steps are (imputer) > features > (polynomial terms) > (scaler) > model.

	The pipeline takes the attributes as they are stored and builds its own feature columns, so the same object
	can be cross validated, fit on every training row and saved, with no preprocessing outside it.

	Args:
		estimator: Final model. It is cloned.
		attributes: Every attribute column the pipeline is given, in the order the feature columns are built.
		shapes: Named shapes per attribute, ex {"Social support": {"shape": HighHinge(t=0.84, keep_raw=True)}}. A
			shape is one transformer, or a list of transformers that are each applied to the attribute and placed
			side by side. Every shape is applied to its own attribute's column only.
		choice: What each attribute contributes: "raw" (the column as it is), "drop" (nothing) or the name of one
			of its shapes. Attributes left out are "raw".
		degree: Degree of the polynomial terms built from the feature columns: every product of up to degree of
			them (ex at 2, each column squared and each pair multiplied), with no constant column. 1 adds nothing.
		expand: Attributes whose feature columns the polynomial terms are built from. The feature columns of the
			other attributes go to the model as they are, after the polynomial terms. None builds the terms from
			every feature column. It has no effect at degree 1.
		scale: Standardize the columns before the model. With degree above 1 the feature columns are also
			standardized before the polynomial terms are built, so a product is not close to a copy of the columns
			it is built from.
		imputer: Fills missing values before the features are built, so no shape sees a missing value. It is
			cloned and must return the attributes in their stored units. Extra columns it adds (ex missing
			indicators) are passed on to the model. None when the pipeline is only given complete rows.

	Returns:
		The unfitted pipeline, with steps "impute" (when imputer is given), "features", "standardize" (when scale
		and degree is above 1), "polynomial" (when degree is above 1), "scale" (when scale) and "model". When
		expand is given and degree is above 1, "standardize" and "polynomial" are inside "features" instead, in
		the pipeline named "expanded".
	"""
	if degree < 1:
		raise ValueError(f"degree must be 1 or more, got {degree}")
	choice = choice or {}
	unknown = [attribute for attribute in choice if attribute not in attributes]
	if unknown:
		raise ValueError(f"choice names attributes that are not in attributes: {unknown}")
	unknown = [attribute for attribute in expand or [] if attribute not in attributes]
	if unknown:
		raise ValueError(f"expand names attributes that are not in attributes: {unknown}")

	def transformers_for(names: list[str]) -> list[tuple]:
		transformers = []
		for attribute in names:
			pick = choice.get(attribute, RAW)
			if pick == RAW:
				transformers.append((attribute, "passthrough", [attribute]))
			elif pick == DROP:
				transformers.append((attribute, "drop", [attribute]))
			else:
				options = shapes.get(attribute, {})
				if pick not in options:
					raise ValueError(f"{attribute} has no shape named {pick!r}, options: {[RAW, DROP] + list(options)}")
				shape = options[pick]
				if isinstance(shape, list):
					transformers += [(f"{attribute} {i}", part, [attribute]) for i, part in enumerate(shape, start=1)]
				else:
					transformers.append((attribute, shape, [attribute]))
		return transformers

	polynomial = []
	if degree > 1:
		if scale:
			polynomial.append(("standardize", StandardScaler()))
		polynomial.append(("polynomial", PolynomialFeatures(degree=degree, include_bias=False)))

	steps = []
	if imputer is not None:
		# the features step selects columns by name, so the imputer has to return a DataFrame
		steps.append(("impute", clone(imputer).set_output(transform="pandas")))
	if polynomial and expand is not None:
		expanded = [attribute for attribute in attributes if attribute in expand]
		others = [attribute for attribute in attributes if attribute not in expand]
		expansion = Pipeline([
			("features", ColumnTransformer(transformers_for(expanded), verbose_feature_names_out=False)),
			*polynomial,
		])
		transformers = [("expanded", expansion, expanded), *transformers_for(others)]
		polynomial = []
	else:
		transformers = transformers_for(attributes)
	# remainder="passthrough" keeps the columns an imputer adds (ex missing indicators)
	steps.append(("features", ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False)))
	steps += polynomial
	if scale:
		steps.append(("scale", StandardScaler()))
	steps.append(("model", clone(estimator)))
	return Pipeline(steps)

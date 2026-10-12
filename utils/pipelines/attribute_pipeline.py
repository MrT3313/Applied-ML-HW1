from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAW = "raw"
DROP = "drop"


def attribute_pipeline(
	estimator: BaseEstimator,
	*,
	attributes: list[str],
	shapes: dict[str, dict[str, BaseEstimator | list[BaseEstimator]]],
	choice: dict[str, str] | None = None,
	scale: bool = False,
	imputer: BaseEstimator | None = None,
) -> Pipeline:
	"""Build one pipeline from a per attribute feature choice and a model: (imputer) > features > (scaler) > model.

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
		scale: Standardize the feature columns before the model.
		imputer: Fills missing values before the features are built, so no shape sees a missing value. It is
			cloned and must return the attributes in their stored units. Extra columns it adds (ex missing
			indicators) are passed on to the model. None when the pipeline is only given complete rows.

	Returns:
		The unfitted pipeline, with steps "impute" (when imputer is given), "features", "scale" (when scale) and
		"model".
	"""
	choice = choice or {}
	unknown = [attribute for attribute in choice if attribute not in attributes]
	if unknown:
		raise ValueError(f"choice names attributes that are not in attributes: {unknown}")

	transformers = []
	for attribute in attributes:
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

	steps = []
	if imputer is not None:
		# the features step selects columns by name, so the imputer has to return a DataFrame
		steps.append(("impute", clone(imputer).set_output(transform="pandas")))
	# remainder="passthrough" keeps the columns an imputer adds (ex missing indicators)
	steps.append(("features", ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False)))
	if scale:
		steps.append(("scale", StandardScaler()))
	steps.append(("model", clone(estimator)))
	return Pipeline(steps)

from sklearn.pipeline import Pipeline


def describe_pipeline(pipeline: Pipeline) -> dict:
	"""Steps and chosen hyperparameters of a fitted SGD pipeline, read back from the fitted object.

	Args:
		pipeline: Fitted pipeline whose last step is an SGD model named "model" (ex from attribute_pipeline).

	Returns:
		{"steps", "degree", "penalty", "alpha", "eta0", "tol", "columns", "epochs run"}. "degree" is 1 when the
		pipeline has no polynomial step, at its top level or nested inside another step. "columns" is the number
		of columns the model is given.
	"""
	model = pipeline["model"]
	degrees = [value for key, value in pipeline.get_params().items() if key.endswith("polynomial__degree")]
	return {
		"steps": " > ".join(pipeline.named_steps),
		"degree": degrees[0] if degrees else 1,
		"penalty": model.penalty,
		"alpha": model.alpha,
		"eta0": model.eta0,
		"tol": model.tol,
		"columns": len(model.coef_),
		"epochs run": model.n_iter_,
	}

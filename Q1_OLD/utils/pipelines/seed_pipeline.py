from sklearn.base import BaseEstimator, clone

from Q1_OLD.consts import SEED


def seed_pipeline(pipeline: BaseEstimator, *, seed: int = SEED) -> BaseEstimator:
	"""Clone a pipeline and set every random_state in it, nested steps included, to seed.

	Args:
		pipeline: Model or pipeline to seed. It is not changed.
		seed: Value given to every random_state parameter.

	Returns:
		The seeded clone.
	"""
	seeded = clone(pipeline)
	params = {
		key: seed
		for key in seeded.get_params(deep=True)
		if key == "random_state" or key.endswith("__random_state")
	}
	if params:
		seeded.set_params(**params)
	return seeded

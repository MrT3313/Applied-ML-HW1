import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from utils.pipelines.make_folds import CVFolds
from utils.pipelines.seed_pipeline import seed_pipeline
from utils.scoring.scorers import root_mean_squared_error


def loss_curve(
	pipeline: Pipeline,
	*,
	X: pd.DataFrame,
	y: pd.Series,
	folds: CVFolds,
	epochs: int = 30,
	seed: int = 42,
) -> pd.DataFrame:
	"""Training and validation rmse of an SGD pipeline after each epoch, on every fold.

	The model is trained one epoch at a time: it is set to max_iter=1 and warm_start=True, so every fit call runs
	one more epoch from the coefficients the previous call ended with. tol=None turns off early stopping, so every
	call runs its one epoch. Each fit call restarts the learning rate schedule at eta0, so with a decaying
	schedule (ex "invscaling") the steps do not keep shrinking from one epoch to the next the way they do inside
	a single fit of many epochs.

	Args:
		pipeline: Pipeline whose last step is an SGD model (ex SGDRegressor). It is cloned and seeded with
			seed_pipeline.
		X: Feature data. Its rows are matched to the folds by index label.
		y: Target values, with the same index as X.
		folds: Folds from make_folds.
		epochs: Number of epochs to run on each fold.
		seed: Value seed_pipeline gives to every random_state in the pipeline.

	Returns:
		One row per fold and epoch (index levels "fold" and "epoch", both counted from 1), with the columns
		"train rmse" and "validation rmse".
	"""
	assert y.index.equals(X.index), "y and X must have the same index"
	assert len(pipeline) > 1, "the pipeline needs at least one step before the model"

	seeded = seed_pipeline(pipeline, seed=seed)
	rows = []
	for fold, (train, validation) in enumerate(folds.positions_for(X), start=1):
		X_train, y_train = X.iloc[train], y.iloc[train]
		X_validation, y_validation = X.iloc[validation], y.iloc[validation]

		# the steps before the model do not change from epoch to epoch, so they are fit once per fold
		preprocess = clone(seeded[:-1]).fit(X_train, y_train)
		Z_train, Z_validation = preprocess.transform(X_train), preprocess.transform(X_validation)

		model = clone(seeded[-1]).set_params(max_iter=1, tol=None, warm_start=True)
		for epoch in range(1, epochs + 1):
			model.fit(Z_train, y_train)
			rows.append({
				"fold": fold,
				"epoch": epoch,
				"train rmse": root_mean_squared_error(y_train, model.predict(Z_train)),
				"validation rmse": root_mean_squared_error(y_validation, model.predict(Z_validation)),
			})
	return pd.DataFrame(rows).set_index(["fold", "epoch"])

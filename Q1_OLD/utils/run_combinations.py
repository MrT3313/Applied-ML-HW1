import hashlib
import json
import time
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from Q1_OLD.consts import SHORT_NAMES
from Q1_OLD.utils.create_pipeline import add_model_to_pipeline, create_preprocessing, preprocessing_labels, step_label
from Q1_OLD.utils.run_experiment import cross_validate_means, seed_model, sort_results

NONE = "none"

# ex: {"right_skew": {"columns": ["Generosity", ...], "transforms": ["log", "sqrt", ...]}, ...}
Groups = dict[str, dict[str, list[str]]]

def skew_aware_combinations(groups: Groups) -> pd.DataFrame:
	# every column picks "none" or one of its group's transforms, which gives the same rows as looping over
	# how many columns change, which columns change, and which transform each one gets.
	# the all-"none" row is dropped, since that is the untransformed baseline (Experiment A)
	options = {col: [NONE, *group["transforms"]] for group in groups.values() for col in group["columns"]}
	combinations = pd.DataFrame(list(product(*options.values())), columns=list(options))
	combinations["n_changed"] = (combinations != NONE).sum(axis=1)
	return combinations[combinations["n_changed"] > 0].reset_index(drop=True)

def combination_code(combination: dict[str, str]) -> str:
	# the "transform" label and saved model file name, ex: "Social=p2,Life exp=yj,Freedom=none,..."
	# (the full ColumnTransformer label would run past 400 characters, over the 255 file name limit)
	return ",".join(f"{SHORT_NAMES.get(col, col)}={code}" for col, code in combination.items())

def combination_pipeline(combination: dict[str, str],
						 *,
						 transforms: dict[str, TransformerMixin],
						 model: RegressorMixin,
						 seed: int
) -> Pipeline:
	# one ColumnTransformer entry per transformed column, so every column gets its own transform;
	# "none" columns and the columns outside the groups pass through unchanged
	entries = [(col, transforms[code], [col]) for col, code in combination.items() if code != NONE]
	preprocessor = create_preprocessing(preprocessing={"transform": ColumnTransformer(entries, remainder="passthrough")})
	return seed_model(add_model_to_pipeline(preprocessor=preprocessor, model=model), seed)

def _fingerprint(*frames: pd.DataFrame | pd.Series) -> str:
	# a short hash of the values, index, and column names, so any change to the data or the matrix shows up
	digest = hashlib.sha256()
	for frame in frames:
		digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
		digest.update(str(list(frame.columns) if isinstance(frame, pd.DataFrame) else frame.name).encode())
	return digest.hexdigest()[:16]

def run_combinations(combinations: pd.DataFrame,
					 *,
					 transforms: dict[str, TransformerMixin],
					 models: dict[str, RegressorMixin],
					 X: pd.DataFrame,
					 y: pd.Series,
					 cv: BaseCrossValidator,
					 scorers: dict,
					 seed: int,
					 csv_path: Path,
					 output_dir: Path,
					 experiment: str,
					 cleaning: str,
					 top_n: int = 10,
					 progress_every: int = 500
) -> pd.DataFrame:
	# same rows and columns as run_experiment, plus one column per transformed feature and n_changed.
	# every result is appended to csv_path as soon as it is scored, and pipelines already in the csv are
	# skipped, so a crashed or interrupted run picks up where it stopped.
	# the settings behind the csv are saved next to it; if any of them change, the csv is deleted and the
	# sweep starts over, so old scores are never mixed with (or mistaken for) new ones
	features = [c for c in combinations.columns if c != "n_changed"]
	records = combinations[features].to_dict("records")
	by_code = {combination_code(combination): combination for combination in records}

	settings = {
		"experiment": experiment,
		"cleaning": cleaning,
		"columns": features,
		"combinations": len(records),
		"combinations_hash": _fingerprint(combinations),
		# step_label names functions and estimators by name, so the label is the same in every session
		"transforms": {code: step_label(transform) for code, transform in transforms.items()},
		"models": {name: step_label(model) for name, model in models.items()},
		"seed": seed,
		"cv": repr(cv),
		"scorers": list(scorers),
		"data_hash": _fingerprint(X, y),
	}
	settings_path = csv_path.with_suffix(".settings.json")
	if csv_path.exists():
		previous = json.loads(settings_path.read_text()) if settings_path.exists() else None
		if previous != settings:
			changed = "no settings were saved with it" if previous is None else \
				"changed: " + ", ".join(k for k in settings if previous.get(k) != settings[k])
			print(f"deleting {csv_path.name} and starting over ({changed})")
			csv_path.unlink()
	settings_path.write_text(json.dumps(settings, indent=2))

	done = set()
	if csv_path.exists():
		previous = pd.read_csv(csv_path)
		done = set(zip(previous["model"], previous["transform"]))

	total = len(records) * len(models)
	start, ran = time.time(), 0
	print(f"{len(done):,} of {total:,} pipelines already in {csv_path.name}")

	for code, combination in by_code.items():
		for model_name, model in models.items():
			if (model_name, code) in done:
				continue
			pipeline = combination_pipeline(combination, transforms=transforms, model=model, seed=seed)
			row = {
				"experiment": experiment,
				"model": model_name,
				"cleaning": cleaning,
				# impute and scale come from the pipeline (both "none" here); transform is the short code
				**preprocessing_labels(pipeline.named_steps["preprocess"]),
				"transform": code,
				**cross_validate_means(pipeline, X=X, y=y, cv=cv, scorers=scorers),
				"path": "",
				**{SHORT_NAMES.get(col, col): combination[col] for col in features},
				"n_changed": sum(choice != NONE for choice in combination.values()),
			}
			pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

			ran += 1
			if ran % progress_every == 0:
				elapsed = time.time() - start
				remaining = total - len(done) - ran
				print(f"{len(done) + ran:,} / {total:,} done, {elapsed / 60:,.1f} min elapsed, "
					  f"~{elapsed / ran * remaining / 60:,.1f} min left")

	results = pd.read_csv(csv_path)
	results["path"] = ""

	# cross_validate throws away its fold models, so refit and save only the best top_n per model
	for model_name, model in models.items():
		for index, code in results[results["model"] == model_name].nsmallest(top_n, "rmse")["transform"].items():
			final_model = clone(combination_pipeline(by_code[code], transforms=transforms, model=model, seed=seed)).fit(X, y)
			model_path = output_dir / f"{model_name}__{code}.joblib"
			joblib.dump(final_model, model_path)
			results.loc[index, "path"] = str(model_path)

	return sort_results(results)

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline

from Q1.utils.complete_row_folds import Folds
from Q1.utils.create_pipeline import add_model_to_pipeline, preprocessing_labels, preprocessing_name

# keyed by (model name, preprocessing name), ex: ("ridge", "scale-StandardScaler")
Pipelines = dict[tuple[str, str], Pipeline]

# most file systems (macOS, Linux) cap a single file name at 255 bytes
MAX_FILE_NAME_BYTES = 255

def model_file_name(model_name: str, preprocessing: str) -> str:
    # "{model}__{preprocessing}.joblib" when it fits, so names that already fit never change.
    # a longer one (ex: IterativeImputer + a ColumnTransformer + a scaler) is cut short and ends with a hash of
    # the full name, so it stays unique and is the same every run; the full settings stay in the results columns
    stem, extension = f"{model_name}__{preprocessing}", ".joblib"
    if len((stem + extension).encode()) <= MAX_FILE_NAME_BYTES:
        return stem + extension

    suffix = f"__{hashlib.sha256(stem.encode()).hexdigest()[:8]}{extension}"
    cut = stem.encode()[:MAX_FILE_NAME_BYTES - len(suffix)].decode(errors="ignore")
    return cut + suffix

def seed_model(pipeline: Pipeline, seed: int) -> Pipeline:
    # seed every model that has a random_state (all except LinearRegression and SVR)
    if "model__random_state" in pipeline.get_params():
        pipeline.set_params(model__random_state=seed)
    return pipeline

def build_pipelines(*,
                    preprocessings: list[Pipeline],
                    models: dict[str, RegressorMixin],
                    seed: int
) -> Pipelines:
    pipelines = {}
    for model_name, model in models.items():
        for preprocessor in preprocessings:
            key = (model_name, preprocessing_name(preprocessor))
            # two preprocessings with the same name would overwrite each other's pipeline and saved model
            assert key not in pipelines, f"duplicate preprocessing {key[1]!r} in this experiment"

            pipelines[key] = seed_model(add_model_to_pipeline(preprocessor=preprocessor, model=model), seed)
    return pipelines

def cross_validate_folds(pipeline: Pipeline,
                         *,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv: BaseCrossValidator | Folds,
                         scorers: dict,
                         return_estimator: bool = False
) -> dict[str, np.ndarray | list]:
    # every score in the project comes from this one cross_validate call
    scores = cross_validate(
        pipeline,         # the model or pipeline to evaluate (cloned and refit on each fold)
        X,                # feature data, split into folds
        y,                # target values, split the same way as X
        scoring=scorers,  # metrics to compute on each validation fold; a dict gives one result per name
        cv=cv,            # a seeded splitter, or a list of (train, validation) folds from complete_row_folds
        n_jobs=-1,        # run folds in parallel on all CPU cores
        return_estimator=return_estimator,  # keep each fold's fitted copy (ex to read a search's best_params_)
    )
    # scores["test_mse"] etc. hold one score per fold, in fold order; flip the sign back to positive
    # (not averaged, not rounded)
    fold_scores = {metric: -scores[f"test_{metric}"] for metric in scorers}
    if return_estimator:
        # one fitted estimator per fold, in the same fold order as the scores
        fold_scores["estimator"] = list(scores["estimator"])
    return fold_scores

def cross_validate_means(pipeline: Pipeline,
                         *,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv: BaseCrossValidator,
                         scorers: dict
) -> dict[str, float]:
    # the same per-fold scores as cross_validate_folds, averaged over the folds (no estimators, so every value
    # is an array of scores)
    fold_scores = cross_validate_folds(pipeline, X=X, y=y, cv=cv, scorers=scorers)
    return {metric: scores.mean() for metric, scores in fold_scores.items()}

def sort_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results
        .sort_values("rmse")
        .reset_index(drop=True)  # renumber rows 0, 1, 2... after sorting
        .round(3)                # trim long decimals
    )

def run_experiment(pipelines: Pipelines,
                   *,
                   X: pd.DataFrame,
                   y: pd.Series,
                   cv: BaseCrossValidator,
                   scorers: dict,
                   output_dir: Path,
                   experiment: str,
                   cleaning: str
) -> pd.DataFrame:
    results = []
    for (model_name, preprocessing), pipeline in pipelines.items():
        metrics = cross_validate_means(pipeline, X=X, y=y, cv=cv, scorers=scorers)

        # cross_validate throws away its fold models, so fit a final copy on all the data and save it
        final_model = clone(pipeline).fit(X, y)
        model_path = output_dir / model_file_name(model_name, preprocessing)
        joblib.dump(final_model, model_path)

        results.append({
            "experiment": experiment,
            "model": model_name,
            # dropping rows happens before the pipeline, so the pipeline can't report it
            "cleaning": cleaning,
            **preprocessing_labels(pipeline.named_steps["preprocess"]),
            **metrics,
            "path": str(model_path),
        })

    return sort_results(pd.DataFrame(results))

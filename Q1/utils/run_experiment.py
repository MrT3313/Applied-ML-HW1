from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline

from Q1.utils.create_pipeline import add_model_to_pipeline, preprocessing_labels, preprocessing_name

# keyed by (model name, preprocessing name), ex: ("ridge", "scale-StandardScaler")
Pipelines = dict[tuple[str, str], Pipeline]

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

def cross_validate_means(pipeline: Pipeline,
                         *,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv: BaseCrossValidator,
                         scorers: dict
) -> dict[str, float]:
    scores = cross_validate(
        pipeline,         # the model or pipeline to evaluate (cloned and refit on each fold)
        X,                # feature data, split into folds
        y,                # target values, split the same way as X
        scoring=scorers,  # metrics to compute on each validation fold; a dict gives one result per name
        cv=cv,            # the same shuffled, seeded folds for every experiment
        n_jobs=-1,        # run folds in parallel on all CPU cores
    )
    # scores["test_mse"] etc. hold one score per fold; average them and flip the sign back to positive
    return {metric: -scores[f"test_{metric}"].mean() for metric in scorers}

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
        model_path = output_dir / f"{model_name}__{preprocessing}.joblib"
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

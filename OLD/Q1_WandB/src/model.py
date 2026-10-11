"""Build the scikit-learn pipeline a config describes. The one place where config values become model code.

To add an option: add its value to the matching table below (or a new step in build_pipeline), then list
that value in a sweep file. A new config key also has to be added in src/config.py: to REQUIRED_KEYS when
every model uses it, or to MODEL_KEYS when only one model does.
"""
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# the values each config key accepts, and what each one builds
# config["data_missing"]: the SimpleImputer strategy to fill with. None means the rows are dropped instead
MISSING = {"drop": None, "impute_median": "median", "impute_mean": "mean"}
SCALERS = {"none": None, "standard": StandardScaler, "minmax": MinMaxScaler}      # config["scaler"]
DUMMY_STRATEGIES = {"mean", "median"}                                             # config["dummy_strategy"]


def build_dummy(config: dict) -> DummyRegressor:
    """The baseline that ignores the features and always predicts the mean or median training target."""
    if config["dummy_strategy"] not in DUMMY_STRATEGIES:
        raise ValueError(
            f"dummy_strategy must be one of {sorted(DUMMY_STRATEGIES)}, got {config['dummy_strategy']!r}"
        )
    return DummyRegressor(strategy=config["dummy_strategy"])


def build_closed_form(config: dict) -> LinearRegression:
    """Least squares linear regression. It has no settings of its own."""
    return LinearRegression()


# config["sgd_penalty"] -> scikit-learn's penalty argument
SGD_PENALTIES = {"none": None, "l2": "l2", "l1": "l1", "elasticnet": "elasticnet"}


def build_sgd(config: dict) -> SGDRegressor:
    """Linear regression fit by stochastic gradient descent, with an optional penalty on large coefficients.

    Closed form solves for the coefficients in one step. SGD reaches them by many small updates, which
    brings a learning rate (sgd_eta0) and makes a penalty possible.

    sgd_penalty picks the kind: "none", "l2" (ridge), "l1" (lasso) or "elasticnet" (a mix of both).
    sgd_alpha is how strong the penalty is, and only matters when there is one. sgd_l1_ratio is the share
    of L1 in the mix, and only matters for elastic net. A setting that does not matter for the chosen
    penalty arrives as None and is not passed on. Everything else keeps scikit-learn's defaults. SGD is
    sensitive to feature scale, so it should be given a scaler.
    """
    if config["sgd_penalty"] not in SGD_PENALTIES:
        raise ValueError(f"sgd_penalty must be one of {sorted(SGD_PENALTIES)}, got {config['sgd_penalty']!r}")

    settings = {"penalty": SGD_PENALTIES[config["sgd_penalty"]], "eta0": config["sgd_eta0"]}
    if config["sgd_penalty"] != "none":
        settings["alpha"] = config["sgd_alpha"]
    if config["sgd_penalty"] == "elasticnet":
        settings["l1_ratio"] = config["sgd_l1_ratio"]
    return SGDRegressor(**settings, random_state=config["seed"])


# config["model_kind"] -> the function that builds that model from the config. A model reads its own
# settings (its MODEL_KEYS in src/config.py) inside its build function
MODELS = {"dummy": build_dummy, "closed_form": build_closed_form, "sgd": build_sgd}


def build_pipeline(config: dict) -> BaseEstimator:
    """Turn a config into the unfitted pipeline it describes: imputer, then scaler, then model.

    The imputer step exists only for the "impute_..." values of data_missing. With "drop" the incomplete
    rows are removed before the pipeline sees the data (usable_rows in src/train.py), so there is nothing
    to fill.

    Because the imputer and scaler are steps of the pipeline, cross validation fits them on each training
    fold alone, so nothing about the validation rows leaks into them.

    Args:
        config: A validated run config. Reads data_missing, scaler, model_kind and that model's own keys.

    Returns:
        The unfitted pipeline.

    Raises:
        ValueError: If one of those keys holds a value that is not supported.
    """
    for key, allowed in (("data_missing", MISSING), ("scaler", SCALERS), ("model_kind", MODELS)):
        if config[key] not in allowed:
            raise ValueError(f"{key} must be one of {sorted(allowed)}, got {config[key]!r}")

    steps = []
    if MISSING[config["data_missing"]] is not None:
        steps.append(SimpleImputer(strategy=MISSING[config["data_missing"]]))
    if SCALERS[config["scaler"]] is not None:
        steps.append(SCALERS[config["scaler"]]())
    steps.append(MODELS[config["model_kind"]](config))
    return make_pipeline(*steps)

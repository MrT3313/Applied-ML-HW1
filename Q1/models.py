from sklearn.base import RegressorMixin
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.svm import SVR, LinearSVR

MODELS: dict[str, RegressorMixin] = {
    "linear_regression": LinearRegression(),
    "sgd": SGDRegressor(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "elastic_net": ElasticNet(),
    # dual=True is required by the default loss and silences the FutureWarning about its default changing;
    # max_iter=10_000 lets it converge on scaled data (the default 1_000 stops early)
    "linear_svr": LinearSVR(dual=True, max_iter=10_000),  # linear model on larger dataset
    # libsvm has no iteration limit by default and never finishes on unscaled squared / cubed features;
    # A to C need at most ~370k iterations, so 1_000_000 only stops the fits that would otherwise hang
    "svr_linear": SVR(kernel="linear", max_iter=1_000_000), # small dataset or nonlinear kernel
}
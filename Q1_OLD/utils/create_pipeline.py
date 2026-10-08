import math
from typing import Literal, TypedDict
from typing_extensions import NotRequired
from html import escape
from IPython.display import HTML, display
from sklearn.utils import estimator_html_repr

from sklearn.base import TransformerMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from Q1_OLD.consts import SHORT_NAMES

Step =Literal["passthrough"] | TransformerMixin

class Preprocessing(TypedDict):
    impute: NotRequired[Step]
    transform: NotRequired[Step]
    scale: NotRequired[Step]

def create_preprocessing(*, preprocessing: Preprocessing) -> Pipeline:
    return Pipeline([
        ("impute", preprocessing.get("impute", "passthrough")),
        ("transform", preprocessing.get("transform", "passthrough")),
        ("scale", preprocessing.get("scale", "passthrough")),
    ])

def add_model_to_pipeline(*,
							preprocessor: Pipeline,
							model: RegressorMixin
) -> Pipeline:
    return Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", clone(model)),
    ])

# the settings this project varies, always shown (even at their default) so every variant of a step reads the same way
ALWAYS_SHOWN = {
    "SimpleImputer": ["strategy"],
    "KNNImputer": ["n_neighbors", "weights"],
    "IterativeImputer": ["estimator", "initial_strategy"],
    "PowerTransformer": ["method"],
    "FunctionTransformer": ["func"],
    "MinShift": ["floor"],
}

# never shown: the seed only picks which random draws a step makes, and feature_names_out only names
# the output columns, so neither changes what the step does
HIDDEN = {"random_state", "feature_names_out"}

def _same(a, b) -> bool:
    # NaN != NaN, so without this the default missing_values=np.nan would always look changed
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        return bool(a == b)
    except (TypeError, ValueError):  # arrays compare elementwise and can't be turned into one bool
        return False

def _format(value) -> str:
    # an estimator inside a step (ex: IterativeImputer's estimator) is named by its class only
    if hasattr(value, "get_params"):
        return type(value).__name__
    # a function (ex: FunctionTransformer's func) is named by its name, not "<function ... at 0x...>"
    if callable(value):
        return value.__name__
    # a dict (ex: FunctionTransformer's kw_args) is flattened, since labels become file names
    if isinstance(value, dict):
        return "(" + ",".join(f"{k}={_format(v)}" for k, v in value.items()) + ")"
    return str(value)

def step_label(step: Step) -> str:
    # "passthrough" means the step is skipped
    if isinstance(step, str):
        return "none"
    # a ColumnTransformer is named by what it does to which columns, with short column names so the label
    # (which is also the saved model's file name) stays under the 255 character limit,
    # ex: "ColumnTransformer(FunctionTransformer(func=log)[Generosity,Neg affect])"
    if isinstance(step, ColumnTransformer):
        parts = [
            f"{step_label(transformer)}[{','.join(SHORT_NAMES.get(c, c) for c in columns)}]"
            for _, transformer, columns in step.transformers
        ]
        # the columns not listed pass through unchanged, unless remainder says otherwise
        if step.remainder != "passthrough":
            # "drop" is a string, and step_label would read any string as "none"
            remainder = step.remainder if isinstance(step.remainder, str) else step_label(step.remainder)
            parts.append(f"remainder={remainder}")
        return f"ColumnTransformer({','.join(parts)})"
    # a Pipeline inside a step runs its steps in order, ex: "MinShift(floor=1)+FunctionTransformer(func=log)"
    if isinstance(step, Pipeline):
        return "+".join(step_label(s) for _, s in step.steps)
    # the varied settings first, then any other setting that differs from the default,
    # ex: "KNNImputer(n_neighbors=10,weights=distance)" or "MinMaxScaler(feature_range=(-1, 1))"
    params = step.get_params(deep=False)
    defaults = type(step)().get_params(deep=False)
    shown = ALWAYS_SHOWN.get(type(step).__name__, [])
    changed = [
        k for k in sorted(params)
        if k not in shown and k not in HIDDEN and not _same(params[k], defaults[k])
    ]
    args = ",".join(f"{k}={_format(params[k])}" for k in [*shown, *changed])
    return f"{type(step).__name__}({args})" if args else type(step).__name__

def preprocessing_labels(preprocessor: Pipeline) -> dict[str, str]:
    # one entry per step, ex: {"impute": "none", "transform": "none", "scale": "StandardScaler"}
    return {name: step_label(step) for name, step in preprocessor.steps}

def preprocessing_name(preprocessor: Pipeline) -> str:
    # skip the "none" steps so names stay short, ex: "impute-SimpleImputer(strategy=median)__scale-StandardScaler"
    used = [f"{name}-{label}" for name, label in preprocessing_labels(preprocessor).items() if label != "none"]
    return "__".join(used) or "no-preprocessing"

def display_pipeline_grid(pipelines: dict, columns: int = 3) -> None:
    cells = "".join(
        f"""
        <div style="border:1px solid #ccc; border-radius:6px; padding:8px; overflow:auto;">
          <div style="font-weight:600; margin-bottom:6px;">{escape(f"{name} | {preprocessing}")}</div>
          {estimator_html_repr(pipeline)}
        </div>
        """
        for (name, preprocessing), pipeline in pipelines.items()
    )
    display(HTML(
        f'<div style="display:grid; grid-template-columns:repeat({columns}, minmax(0, 1fr)); gap:12px;">'
        f"{cells}</div>"
    ))
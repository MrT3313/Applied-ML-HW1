# /src

> Pure python, no `wandb`

The machine learning code. Nothing here knows that W&B exists: functions take a config dict and data, and
return results. That keeps this code testable by itself and lets the tracker be swapped without touching it.

| Module | What it does |
| --- | --- |
| `config.py` | Loads a YAML config and checks it has exactly the expected keys |
| `data.py` | Reads a dataset CSV into features and target, and selects complete rows |
| `model.py` | Builds the scikit-learn pipeline (imputer, scaler, model) that a config describes |
| `evaluate.py` | Scores a pipeline with cross validation, or a fitted model on one dataset |
| `train.py` | Ties those together: `train()` scores a config, `fit()` produces the model to keep |
| `compare.py` | The noise rule: is a run really better than the baseline, or within fold to fold noise? |
| `grid.py` | Expands a sweep file into the exact list of runs, honoring its `when` conditions |
| `utils/make_folds.py` | Builds the cross validation folds, so every run is validated on the same rows |
| `utils/scorers.py` | The metrics (rmse, mae, r2). Add one here and it is reported everywhere |
| `utils/standard_error.py` | Corrected standard error of a paired per fold difference |

How one training run flows through it:

```text
config --> build_pipeline --+
data ----> make_folds ------+--> evaluate --> summarize --> TrainResult
           usable_rows -----+
```

To add an option to search over, see the header of `model.py`.

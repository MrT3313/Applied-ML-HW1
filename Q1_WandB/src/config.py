"""Load and check the config: the flat dict of settings that fully describes one run.

A run's config is the "run" section of configs/base.yaml with the sweep's parameters laid over it. W&B
records every key on the run, which is what lets the UI filter, group and compare runs by their settings.

To add a setting every model uses: add its key to REQUIRED_KEYS, read it somewhere in src/, and give it
values in every sweep file. To add a setting only one model uses: add its key to MODEL_KEYS instead, and
only that model's sweep files have to set it.
"""
from pathlib import Path

import yaml

# the keys every run's config must have, whatever the model
REQUIRED_KEYS = {
	# the same for every run, from configs/base.yaml
	'data_train',                            # dataset artifact to train on, ex "happiness-train:v0"
	'cv_seeds', 'cv_n_splits',               # the folds: one shuffled KFold repeat per seed
	'seed',                                  # random_state for models with randomness of their own, ex SGD
	# chosen by each sweep, from the sweep file's parameters
	'data_missing', 'scaler',                # preprocessing, see MISSING and SCALERS in src/model.py
	'model_kind'                             # which model to build, see MODELS in src/model.py
}

# the extra keys each model_kind needs on top of REQUIRED_KEYS. A run must have its own model's keys and
# none of the other models', so a closed form sweep never has to mention a dummy setting
MODEL_KEYS = {
	'dummy': {'dummy_strategy'},             # "mean" or "median" of the training targets
	'closed_form': set(),
	'sgd': {
		'sgd_penalty',                       # "none", "l2" (ridge), "l1" (lasso) or "elasticnet" (a mix of both)
		'sgd_alpha',                         # strength of the penalty. None when there is no penalty
		'sgd_l1_ratio',                      # share of L1 in an elastic net penalty. None for the other penalties
		'sgd_eta0',                          # starting learning rate
	},
}

def load_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate(config: dict) -> dict:
    """Fail on a missing or an unknown key. Returns the config unchanged.

    The expected keys are REQUIRED_KEYS plus the MODEL_KEYS of the config's own model_kind. A typo in a
    sweep file then stops the run instead of silently training with something else. The values themselves
    are checked later, by build_pipeline. A key may hold None, which means it does not apply to this run
    (see "when" in src/grid.py).
    """
    expected = REQUIRED_KEYS | MODEL_KEYS.get(config.get("model_kind"), set())
    missing = expected - config.keys()
    unknown = config.keys() - expected
    if missing or unknown:
        raise ValueError(f"config keys missing: {sorted(missing)}, unknown: {sorted(unknown)}")

    return config
"""Create a sweep on the W&B server from a sweep file and print its path.

A sweep is a list of runs to do, stored on the server. Creating a sweep runs nothing. An agent
(`wandb agent <path>`) then asks the server for one run at a time and launches entrypoints/run.py for each.

The definition sent to the server is two files merged:
    configs/base.yaml, section "sweep"   what every sweep shares (the program to launch, the metric)
    the sweep file given as argument     what this experiment decides (name, parameters)

The sweep file's parameters are not handed to W&B as they are. A W&B grid crosses every parameter with
every other and cannot skip one that does not apply, so src/grid.py expands the file into the exact list
of runs first, honoring its "when" conditions. W&B receives that list as a single parameter, "combo", whose
values are the runs, each as a JSON string. run.py unpacks it back into ordinary config keys.

Usage, from inside Q1_WandB. The run-sweep target calls this, then starts the agent, then compares:
    python -m entrypoints.sweep configs/sweeps/linear_regression/closed_form.yaml

The last line printed is the sweep's full path (entity/project/sweep_id), which the Makefile hands to the
agent.
"""
import json
import sys
from pathlib import Path

import wandb
import yaml

from src.grid import COMBO_KEY, expand

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "happiness"

base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
experiment = yaml.safe_load(Path(sys.argv[1]).read_text())

# wandb.Api() reads what is already on the server. entity is the account the project lives under
api = wandb.Api()
entity = api.default_entity

# every run of the sweep would fail without its dataset, so check before the sweep is created
data_train = base["run"]["data_train"]
try:
    api.artifact(f"{entity}/{PROJECT}/{data_train}")
except wandb.errors.CommError:
    sys.exit(f"{data_train} is not on the server. Run the register-data-artifacts target, then set "
             f"data_train in configs/base.yaml to the version it created.")

if experiment.get("method") != "grid":
    sys.exit("only 'method: grid' is supported, because the list of runs is built here and not by W&B")

# one config dict per run. The first one is the sweep's baseline (see entrypoints/compare.py)
combos = expand(experiment["parameters"])
print(f"{experiment['name']}: {len(combos)} runs", file=sys.stderr)

# the experiment file wins where both define a key. Its parameters are replaced by the expanded list
definition = {**base["sweep"], **experiment}
definition["parameters"] = {COMBO_KEY: {"values": [json.dumps(combo) for combo in combos]}}
sweep_id = wandb.sweep(definition, entity=entity, project=PROJECT)
# the full path, because the agent on a local server needs the entity as well as the project
print(f"{entity}/{PROJECT}/{sweep_id}")

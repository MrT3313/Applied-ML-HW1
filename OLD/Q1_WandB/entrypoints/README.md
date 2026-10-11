# Entrypoints

> The _only_ files that import `wandb`

Each script is one step of the workflow. It talks to the W&B server, calls into `src/` for the actual work,
and records the result. Every script that opens a run labels it with a `job_type`, which is how the steps
are told apart in the UI.

| Script | Make target | `job_type` | Reads from W&B | Writes to W&B |
| --- | --- | --- | --- | --- |
| `register_data.py` | `register-data-artifacts` | `data` | nothing | The `happiness-train` and `happiness-test` dataset artifacts |
| `sweep.py` | part of `run-sweep` | opens no run | Checks the dataset exists | A sweep (the search definition) |
| `run.py` | started by the agent in `run-sweep` | `train` | `happiness-train` | One run per configuration: config, per fold scores, summary scores |
| `compare.py` | part of `run-sweep`, or `compare` | opens no run | Per fold scores of every run in a sweep | `verdict` and `delta_*` columns on each of those runs |
| `promote.py` | `promote` | `promote` | One run's config, `happiness-train` | The `happiness-regressor` model artifact, alias `candidate` |
| `final_eval.py` | `final-eval` | `test` | `happiness-regressor`, `happiness-test` | Test scores |

Run them through the Makefile, from inside `Q1_WandB`. Scripts that import `src` only work when started
from that folder with `python -m entrypoints.<name>`.

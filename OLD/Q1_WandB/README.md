# Q1 with Weights & Biases

The Q1 happiness regression, rebuilt so that experiments are defined in config files, run with one command,
and compared in the W&B UI. `Project.md` holds the rules of the project (metric, holdout, promotion rule).
This file explains how the pieces work.

## W&B in seven terms

| Term | What it is | Where it comes from here |
| --- | --- | --- |
| Run | One tracked execution of a script | Every `wandb.init(...)` in `entrypoints/` |
| Config | A run's inputs. Shown as table columns you can filter and sort by | `configs/base.yaml` plus the sweep's parameters |
| History | Values a run logs step by step. Drawn as line charts | The per fold scores, `fold/rmse` and friends |
| Summary | One final value per key per run. Shown as table columns | `rmse_mean`, `verdict`, `delta_rmse`, ... |
| Artifact | A named, versioned file on the server (`name:v0`, `name:v1`) | The two datasets and the promoted model |
| Sweep | A search definition: which config keys to vary and how | A file in `configs/sweeps/` |
| Agent | The process that asks a sweep for the next combination and launches a run for it | Started by `make run-sweep` |

Lineage is the graph W&B builds from which runs read and wrote which artifacts. It is how you can check
that only the final evaluation ever touched the test set.

## Layout

| Path | Contents |
| --- | --- |
| `configs/base.yaml` | Settings shared by every run and every sweep |
| `configs/sweeps/` | One file per experiment: the options to search over |
| `entrypoints/` | The scripts that talk to W&B, one per workflow step. See its README |
| `src/` | The machine learning code, with no W&B in it. See its README |
| `DATA/` | The CSVs that get uploaded as dataset artifacts |
| `Makefile.WeightsAndBiases` | Every step as a make target |
| `wandb/`, `artifacts/`, `MODELS/` | Local caches W&B and the scripts create. Git ignored, safe to delete |

## Workflow

Run everything from inside `Q1_WandB`. The local server must be up (`wandb server start`).

| Step | Command | How often |
| --- | --- | --- |
| 1. Upload the data | `make -f Makefile.WeightsAndBiases register-data-artifacts` | Once |
| 2. Run an experiment | `make -f Makefile.WeightsAndBiases run-sweep` | Per experiment |
| 3. Decide | Read the sweep's runs table in the UI | Per experiment |
| 4. Save the chosen model | `make -f Makefile.WeightsAndBiases promote RUN=<run_id>` | When a run is chosen |
| 5. Score it on the test set | `make -f Makefile.WeightsAndBiases final-eval` | Once, at the end |

Step 2 does three things in a row: creates the sweep, runs every combination in its file, and writes each
run's comparison against the baseline onto the run. Use `SWEEP=<file>` to run a different sweep file.

## Reading the result (step 3)

Open the sweep URL printed at the end of step 2 and look at its runs table. Runs are named after their
choices, ex `model_kind=closed_form | data_missing=impute_mean | scaler=standard`. Add these columns with the Columns button if they are hidden.

| Column | Meaning |
| --- | --- |
| `rmse_mean` | Mean validation RMSE over the 20 folds. Lower is better |
| `verdict` | `baseline`, `identical`, `within noise`, `candidate`, or `worse` |
| `delta_rmse` | Mean per fold RMSE difference from the baseline. Negative is better |
| `delta_over_se` | That difference in standard errors. Below -1 is a real improvement |

A `candidate` beats the baseline by more than noise and is worth promoting. If there is none, the baseline
stays, because within noise the simpler model wins. The baseline is the sweep's first run, which is the
first value of every list in the sweep file.

## Adding an option to search over

1. Add the key and its values under `parameters` in a sweep file.
2. Add the key in `src/config.py`: to `REQUIRED_KEYS` if every model uses it (then every sweep file must
   set it), or to that model's entry in `MODEL_KEYS` if only one model does (ex `dummy_strategy`).
3. Read `config["<key>"]` where it takes effect: `build_pipeline` in `src/model.py` for a shared option, the
   model's own build function for a model only one.

Each key with several values multiplies the number of runs in a grid.

## Options that only apply sometimes (`when`)

Some settings only mean something under another choice: a penalty strength needs a penalty. Add `when` to
such a parameter and it is only varied in the runs where the condition holds. Everywhere else it is `None`
and no model is trained twice.

```yaml
sgd_alpha:
  values: [0.001, 0.01, 0.1]
  when: {sgd_penalty: [l2, l1, elasticnet]}
```

The parameter named in `when` must be defined above it in the file. W&B grids cannot do this themselves, so
`entrypoints/sweep.py` builds the list of runs (`src/grid.py`) and W&B only hands them out. That is why a
sweep's own parameter list in the UI shows a single entry, `combo`. Each run still carries every setting as
an ordinary config column.

## Things that will trip you up

- **Deleting the `data` run can delete the datasets.** Register again, then make `data_train` in
  `configs/base.yaml` and `DATA_TEST` in `entrypoints/final_eval.py` match the new version number.
- **Sweeps can only be deleted in the UI**, not from code.
- **The workspace layout is saved in the UI.** Charts from older runs stay until you remove them.
- **`final-eval` is not for choosing.** Pick the model from the cross validation comparison, then score it once.

"""Expand a sweep file's parameters into the exact list of runs, honoring "when" conditions.

A W&B grid always crosses every parameter with every other one. It cannot skip a parameter that is
irrelevant for some choices, ex a penalty strength when there is no penalty, and so it would run the same
model several times. This module does the crossing itself instead:

    sgd_alpha:
      values: [0.001, 0.01, 0.1]
      when: {sgd_penalty: [l2, l1, elasticnet]}

means "try these values only in runs whose sgd_penalty is one of those". In every other run the key is
set to None and is not varied. entrypoints/sweep.py sends the resulting list to W&B, which then only
hands the runs out one at a time.
"""

# the single W&B sweep parameter that carries one expanded combination, as JSON, to each run
COMBO_KEY = "combo"


def expand(parameters: dict) -> list[dict]:
    """Turn a sweep file's parameters block into one config dict per run.

    Each parameter holds either "value" (one fixed setting) or "values" (settings to try), and optionally
    "when": a dict of earlier parameter -> the values of it under which this parameter applies.

    Args:
        parameters: The "parameters" block of a sweep file, in file order.

    Returns:
        Every valid combination, in the order a grid walks them: the first value of every parameter first,
        the last parameter varying fastest. A parameter whose "when" does not hold is None in that
        combination.

    Raises:
        ValueError: If a "when" names a parameter that is not defined above it in the file.
    """
    combos = [{}]
    for key, spec in parameters.items():
        values = spec["values"] if "values" in spec else [spec["value"]]
        # a condition may be written as one value or as a list of values
        when = {name: allowed if isinstance(allowed, list) else [allowed] for name, allowed in spec.get("when", {}).items()}
        unknown = [name for name in when if name not in combos[0]]
        if unknown:
            raise ValueError(f"{key}: 'when' refers to {unknown}, which must be defined above it in the file")

        expanded = []
        for combo in combos:
            if all(combo[name] in allowed for name, allowed in when.items()):
                expanded.extend({**combo, key: value} for value in values)
            else:
                expanded.append({**combo, key: None})
        combos = expanded
    return combos

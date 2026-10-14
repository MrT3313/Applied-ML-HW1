from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression


def shape_r2_df(
	*,
	df: pd.DataFrame,
	target: str,
	shapes: dict[str, dict[str, pd.DataFrame]],
	margin: float,
	control: Optional[list[str]] = None,
) -> pd.DataFrame:
	"""Score every shape of every attribute by its R² with the target, and mark the shapes to carry forward.

	Each shape is scored by the R² of a linear regression of the target on the shape's columns. The first shape
	of each attribute is its baseline. Another shape is carried when its R² beats the baseline's by at least
	`margin`. The baseline is carried only when no other shape of that attribute is.

	R² is measured on the same rows the fit used, and the thresholds were searched on those rows too, so every
	gain is optimistic.

	Args:
		df: DataFrame holding the target and the control columns, on the same index as the shapes.
		target: Column the shapes are meant to predict.
		shapes: Attribute name to that attribute's shapes (shape name to DataFrame of columns).
		margin: Smallest R² gain over the baseline for a shape to be carried.
		control: Columns of `df` added to every fit, the baseline included. None adds no column.

	Returns:
		A DataFrame with one row per attribute and shape, indexed by `attribute` and `shape`, attributes in the
		order given, shapes sorted by r2 within each attribute, highest first:
			columns: number of columns the shape has, control columns not counted.
			r2: R² of the shape's columns plus the control columns, rounded to 3 decimals.
			gain: r2 minus the r2 of the attribute's baseline, rounded to 3 decimals.
			carried: whether the shape is carried forward, decided before rounding.
	"""
	control = control or []
	tables = {}
	for attribute, attribute_shapes in shapes.items():
		rows = {}
		for name, shape in attribute_shapes.items():
			# rows where the shape, the control columns and the target are all present
			complete = shape.join(df[control + [target]], how='inner').dropna()
			X = complete.drop(columns=target)
			y = complete[target]
			rows[name] = {'columns': shape.shape[1], 'r2': LinearRegression().fit(X, y).score(X, y)}

		table = pd.DataFrame.from_dict(rows, orient='index')
		table['gain'] = table['r2'] - table['r2'].iloc[0]  # the first shape is the baseline

		carried = (table['gain'] >= margin).to_numpy()
		carried[0] = not carried[1:].any()  # the baseline is carried only when no other shape is
		table['carried'] = carried

		tables[attribute] = table.sort_values('r2', ascending=False)

	return pd.concat(tables, names=['attribute', 'shape']).round({'r2': 3, 'gain': 3})

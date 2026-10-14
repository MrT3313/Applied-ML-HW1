import numpy as np
import pandas as pd

from utils.statistics.find_threshold import find_threshold


def continuous_shapes(*, df: pd.DataFrame, feature: str, target: str) -> dict[str, pd.DataFrame]:
	"""Build the shapes of one continuous attribute: the ways of feeding it to a straight line fit.

	A shape is a DataFrame of one or more columns built from the attribute. Two thresholds are searched on the
	rows of `df`, one for a hinge and one for a step, and each is written into the names of the shapes that use it.
	The first shape, 'raw', is the baseline the other shapes are compared against.

	Args:
		df: DataFrame holding the attribute and the target.
		feature: Attribute the shapes are built from.
		target: Column the attribute is meant to predict. Only used to search the thresholds.

	Returns:
		A dict of shape name to DataFrame, each indexed by the rows where the attribute is present:
			raw: the attribute as it is.
			flat above t: the attribute stops counting above t.
			flat below t: the attribute only counts above t.
			raw + hinge at t: the slope changes at t.
			step alone at t: one level below t, another at or above it.
			raw + step at t: a straight line that jumps at t.
	"""
	# drop the missing rows before building the step column, otherwise a missing value would become 0
	x = df[feature].dropna()

	hinge_t = find_threshold(df=df, feature=feature, target=target, kind='hinge').threshold
	step_t = find_threshold(df=df, feature=feature, target=target, kind='step').threshold
	hinge_name = f'{round(hinge_t, 3):g}'
	step_name = f'{round(step_t, 3):g}'

	hinge = (x - hinge_t).clip(lower=0).rename('hinge')  # how far the attribute is past t, 0 below it
	step = (x >= step_t).astype(float).rename('step')  # 0 below t, 1 at or above it

	return {
		'raw': x.to_frame(),
		f'flat above {hinge_name}': x.clip(upper=hinge_t).to_frame(),
		f'flat below {hinge_name}': x.clip(lower=hinge_t).to_frame(),
		f'raw + hinge at {hinge_name}': pd.concat([x, hinge], axis=1),
		f'step alone at {step_name}': step.to_frame(),
		f'raw + step at {step_name}': pd.concat([x, step], axis=1),
	}


def categorical_shapes(*, df: pd.DataFrame, feature: str) -> dict[str, pd.DataFrame]:
	"""Build the shapes of one categorical attribute: the ways of feeding it to a straight line fit.

	The first shape, 'code as number', is the baseline the other shape is compared against.

	Args:
		df: DataFrame holding the attribute.
		feature: Attribute the shapes are built from. Its levels are stored as numbers.

	Returns:
		A dict of shape name to DataFrame, each indexed by the rows where the attribute is present:
			code as number: the level's number in one column, as if the levels were evenly spaced steps.
			one-hot: one 0/1 column per level, minus the first level, which the intercept covers.
	"""
	x = df[feature].dropna()

	return {
		'code as number': x.to_frame(),
		'one-hot': pd.get_dummies(x, prefix=feature, drop_first=True, dtype=float),
	}


def sine_cosine(*, series: pd.Series, period: int) -> pd.DataFrame:
	"""Turn a repeating attribute into a sine column and a cosine column.

	It places the levels on a circle, so the last level sits next to the first.

	Args:
		series: Attribute whose levels repeat, ex Month.
		period: Number of levels in one full cycle, ex 12 for Month.

	Returns:
		A DataFrame with the columns 'sine' and 'cosine', on the same index as the series.
	"""
	angle = 2 * np.pi * series / period

	return pd.DataFrame({'sine': np.sin(angle), 'cosine': np.cos(angle)})

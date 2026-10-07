from pandas import DataFrame, Series, concat

from typing import Optional

from utils.rule_of_thumb__skewness import rule_of_thumb__skewness

from .bucket import bucket_transform
from .log import log_transform
from .power import power_transform
from .rank import rank_transform
from .sklearn_wrapper import SeriesTransformer
from .sqrt import sqrt_transform
from .yeo_johnson import yeo_johnson_transform

# Transforms that reshape a numeric column but keep the distances between values, so a linear
# model's slope still means something. These are the fair candidates for fixing skew
NUMERIC_TRANSFORMS = {
	"log": lambda s: log_transform(series=s),
	"sqrt": lambda s: sqrt_transform(series=s),
	"power_0.25": lambda s: power_transform(series=s, power=0.25),
	# "power_0.50": lambda s: power_transform(series=s, power=0.50), # same as sqrt
	"power_0.75": lambda s: power_transform(series=s, power=0.75),
	"yeo_johnson": lambda s: yeo_johnson_transform(series=s),
}

# Every available transform by name, so callers can loop over all of them. Buckets are for one-hot
# encoding and rank is for order-only features, so neither competes in compare_all_transforms
TRANSFORMS = {
	**NUMERIC_TRANSFORMS,
	'bucket_quantile': lambda s: bucket_transform(series=s, strategy="quantile", n_buckets=10),
	'bucket_uniform': lambda s: bucket_transform(series=s, strategy='uniform', n_buckets=10),
	'bucket_kmeans': lambda s: bucket_transform(series=s, strategy='kmeans', n_buckets=10),
	"rank": lambda s: rank_transform(series=s),
}

SKEW_FIXED_BELOW = 0.5

def compare_all_transforms(*, series: Series, target: Optional[Series] = None) -> DataFrame:
	"""
		Does:
			Runs every NUMERIC_TRANSFORMS entry on series and reports, per transform:
			  #2 does it fix the skew (|skew| < 0.5)
			  #3 does it make the relationship with target more linear (higher |Pearson corr|)
		Rows:
			"none" is the untransformed series, so every other row can be compared against it
		Sort:
			Transforms that fix the skew first, then the biggest corr gain (or smallest |skew| without a target)
	"""
	candidates = {"none": lambda s: s, **NUMERIC_TRANSFORMS}
	rows = []
	for name, transform in candidates.items():
		# Transforms whose domain doesn't fit the data (ex: log on negatives) assert, so keep the row with NaN
		try:
			transformed = transform(series)
		except AssertionError:
			rows.append({"transform": name, "skew_rule": "not allowed for this data", "fixes_skew": False})
			continue

		row = {
			"transform": name,
			"skew": transformed.skew(),
			"skew_rule": rule_of_thumb__skewness(value=transformed.skew()),
			"fixes_skew": abs(transformed.skew()) < SKEW_FIXED_BELOW,
			"kurt": transformed.kurt(),
		}
		if target is not None:
			# Series.corr aligns on the index and drops rows where either side is NaN
			row["corr_with_target"] = transformed.corr(target)
		rows.append(row)

	summary = DataFrame(rows).set_index("transform")

	if target is None:
		return summary.sort_values(["fixes_skew", "skew"], ascending=[False, True], key=_abs_if_numeric, na_position="last")

	# Compare strength, not sign, so a negative relationship (ex: Negative affect) still counts as linear
	summary["corr_gain"] = summary["corr_with_target"].abs() - abs(summary.loc["none", "corr_with_target"])
	return summary.sort_values(["fixes_skew", "corr_gain"], ascending=[False, False], na_position="last")


def compare_all_transforms_for_df(*, df: DataFrame, target: str) -> DataFrame:
	"""
		Does:
			Runs compare_all_transforms on every column of df except target, stacked into one DataFrame
			indexed by (feature, transform). The top row of each feature is the best candidate

		Example:
			skewed = [c for c in features if rule_of_thumb__skewness(series=train_df[c]) == "Highly skewed"]
			compare_all_transforms_for_df(df=train_df[skewed + [TARGET]], target=TARGET)
	"""
	assert target in df.columns, f"df must include the target column {target!r} (ex: df[features + [target]])"
	features = [c for c in df.columns if c != target]
	per_feature = [compare_all_transforms(series=df[c], target=df[target]) for c in features]
	return concat(per_feature, keys=features, names=["feature", "transform"])


def _abs_if_numeric(s: Series) -> Series:
	return s.abs() if s.dtype.kind == "f" else s


__all__ = [
	"SeriesTransformer",
	"NUMERIC_TRANSFORMS",
	"TRANSFORMS",
	"compare_all_transforms",
	"compare_all_transforms_for_df",

	"bucket_transform",
	"log_transform",
	"power_transform",
	"rank_transform",
	"sqrt_transform",
	"yeo_johnson_transform",
]

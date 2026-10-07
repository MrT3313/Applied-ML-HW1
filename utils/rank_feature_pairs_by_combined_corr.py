from itertools import combinations

import pandas as pd


def rank_feature_pairs_by_combined_corr(*, df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rank every pair of features by how much better they predict the target together than alone.

    The combined correlation is the multiple correlation R of a linear regression of the target
    on both features, computed directly from the pairwise correlations (no model is fit).

    Args:
        df: DataFrame holding the target and the numeric features.
        target: Column to predict. Every other column is treated as a feature.

    Returns:
        A DataFrame with one row per feature pair, sorted by gain_from_combining (highest first).
    """
    correlation_matrix = df.corr()
    features = correlation_matrix.columns.drop(target)

    rows = []
    for feature_a, feature_b in combinations(features, 2):
        feature_a_corr_to_target = correlation_matrix.loc[feature_a, target]
        feature_b_corr_to_target = correlation_matrix.loc[feature_b, target]
        feature_a_corr_to_feature_b = correlation_matrix.loc[feature_a, feature_b]

        # how well feature_a and feature_b predict the target when used together (multiple correlation R)
        combined_r_squared = (
            feature_a_corr_to_target**2
            + feature_b_corr_to_target**2
            - 2 * feature_a_corr_to_target * feature_b_corr_to_target * feature_a_corr_to_feature_b
        ) / (1 - feature_a_corr_to_feature_b**2)
        combined_corr_to_target = combined_r_squared ** 0.5

        best_single_corr_to_target = max(abs(feature_a_corr_to_target), abs(feature_b_corr_to_target))

        rows.append({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_a_corr_to_feature_b": feature_a_corr_to_feature_b,
            "best_single_corr_to_target": best_single_corr_to_target,
            "combined_corr_to_target": combined_corr_to_target,
            "gain_from_combining": combined_corr_to_target - best_single_corr_to_target,
        })

    return pd.DataFrame(rows).sort_values("gain_from_combining", ascending=False, ignore_index=True)

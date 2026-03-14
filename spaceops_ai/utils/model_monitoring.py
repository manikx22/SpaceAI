"""Model monitoring helpers: drift score and status classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    """Drift monitoring summary."""

    score: float
    status: str
    feature_scores: dict[str, float]


def baseline_stats(df: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return baseline means/stds from reference telemetry."""
    means = df[features].mean()
    stds = df[features].std(ddof=0).replace(0, 1e-6)
    return means, stds


def compute_drift_report(
    baseline_means: pd.Series,
    baseline_stds: pd.Series,
    current_df: pd.DataFrame,
    features: list[str],
    warn_threshold: float,
    critical_threshold: float,
) -> DriftReport:
    """Compute normalized drift score from rolling/current telemetry."""
    if current_df.empty:
        return DriftReport(score=0.0, status="NOMINAL", feature_scores={f: 0.0 for f in features})

    recent = current_df[features].tail(min(40, len(current_df)))
    recent_means = recent.mean()

    z_scores = ((recent_means - baseline_means).abs() / baseline_stds).clip(lower=0.0)
    # Map z-score to bounded drift score so it is dashboard-friendly.
    feature_scores = {f: float(np.clip(z_scores[f] / 3.0, 0.0, 1.0)) for f in features}
    score = float(np.mean(list(feature_scores.values())))

    if score >= critical_threshold:
        status = "CRITICAL"
    elif score >= warn_threshold:
        status = "WARNING"
    else:
        status = "NOMINAL"

    return DriftReport(score=score, status=status, feature_scores=feature_scores)

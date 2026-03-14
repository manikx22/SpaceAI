"""Data quality checks for telemetry ingestion and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    """Structured quality results for telemetry frames."""

    total_rows: int
    null_cells: int
    duplicate_rows: int
    out_of_range_cells: int
    quality_score: float

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "null_cells": self.null_cells,
            "duplicate_rows": self.duplicate_rows,
            "out_of_range_cells": self.out_of_range_cells,
            "quality_score": self.quality_score,
        }


def build_quality_report(df: pd.DataFrame, value_columns: list[str]) -> DataQualityReport:
    """Compute basic quality metrics for telemetry table."""
    total_rows = int(len(df))
    if total_rows == 0:
        return DataQualityReport(0, 0, 0, 0, 0.0)

    null_cells = int(df[value_columns].isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    out_of_range = int(((df[value_columns] < 0.0) | (df[value_columns] > 1.0)).sum().sum())

    row_penalty = (null_cells + duplicate_rows + out_of_range) / max(1.0, float(total_rows))
    quality_score = float(np.clip(100.0 - row_penalty * 100.0, 0.0, 100.0))

    return DataQualityReport(
        total_rows=total_rows,
        null_cells=null_cells,
        duplicate_rows=duplicate_rows,
        out_of_range_cells=out_of_range,
        quality_score=quality_score,
    )


def sanitize_frame(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Clean and clamp telemetry values for robust downstream training."""
    clean = df.copy()
    clean = clean.drop_duplicates()
    clean[value_columns] = clean[value_columns].replace([np.inf, -np.inf], np.nan)
    clean[value_columns] = clean[value_columns].interpolate(limit_direction="both")
    clean[value_columns] = clean[value_columns].fillna(clean[value_columns].median())
    clean[value_columns] = clean[value_columns].clip(lower=0.0, upper=1.0)
    return clean.reset_index(drop=True)

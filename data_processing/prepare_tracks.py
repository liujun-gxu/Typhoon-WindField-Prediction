"""Minimal best-track cleaning and year-based train/validation/test splitting."""

from __future__ import annotations

import pandas as pd


def split_from_year(year: int) -> str:
    if year in (2020, 2021):
        return "train"
    if year == 2022:
        return "val"
    if year == 2023:
        return "test"
    raise ValueError(f"Expected a study year in 2020--2023, got {year}")


def prepare_hourly_track_table(frame: pd.DataFrame, time_column: str = "time") -> pd.DataFrame:
    """Validate a prepared hourly CMA best-track table and attach the fixed split."""
    required = {"storm_uid", time_column, "latitude", "longitude"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Track table missing columns: {sorted(missing)}")
    output = frame.copy()
    output[time_column] = pd.to_datetime(output[time_column], utc=False)
    if output[["latitude", "longitude"]].isna().any().any() or output.duplicated(["storm_uid", time_column]).any():
        raise ValueError("Track table contains missing coordinates or duplicate storm/time records")
    output["split"] = output[time_column].dt.year.map(split_from_year)
    return output.sort_values(["storm_uid", time_column]).reset_index(drop=True)

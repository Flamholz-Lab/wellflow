import pandas as pd
import numpy as np


def add_blank_correction(df: pd.DataFrame, window: int = 4, od_col: str = "od") -> pd.DataFrame:
    """Add a blank-corrected OD column ('od_blank').

    Computes a per-well blank by averaging the first ``window`` timepoints,
    then subtracts it from every measurement. Values that go negative are
    clipped to zero.

    Args:
        df: Measurements with 'well' and 'time_hours' columns.
        window: Number of initial timepoints to average for the blank.
        od_col: Name of the raw OD column.

    Returns:
        Copy of df with an added 'od_blank' column.
    """
    df = df.sort_values(["well", "time_hours"]).copy()
    df["od_blank"] = np.nan
    for well, group in df.groupby("well", sort=False):
        blank_value = group[od_col].iloc[:window].mean()
        blanked = (group[od_col] - blank_value).clip(lower=0)
        df.loc[group.index, "od_blank"] = blanked.to_numpy()
    df = df.sort_values(["time", "well"]).reset_index(drop=True)
    return df


def add_smoothed_od(
    df: pd.DataFrame,
    group_by: str | list[str] = "well",
    od_col: str = "od_blank",
    window: int = 5,
) -> pd.DataFrame:
    """Add a rolling-average smoothed OD column ('od_smooth').

    Args:
        df: Measurements with 'time_hours' and grouping column(s).
        group_by: Column(s) to group by when smoothing.
        od_col: Name of the OD column to smooth.
        window: Rolling window size.

    Returns:
        Copy of df with an added 'od_smooth' column.
    """
    sort_by = ["time_hours"] + (group_by if isinstance(group_by, list) else [group_by])
    df = df.copy()
    df["od_smooth"] = np.nan
    df = df.sort_values(by=sort_by)
    for _, group in df.groupby(group_by):
        smoothed = group[od_col].rolling(window, center=True, min_periods=1).mean()
        df.loc[group.index, "od_smooth"] = smoothed
    df.sort_values(by=sort_by, inplace=True)
    return df

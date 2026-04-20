import pandas as pd
from pathlib import Path


def read_flagged_wells(path: str, well_col: str = "well", desc_well: str = "notes") -> pd.DataFrame:
    """Read a table of flagged wells and return a normalized DataFrame.

    Args:
        path: Path to an .xlsx or .csv file with at least a well column.
        well_col: Column name for well identifiers in the file.
        desc_well: Column name for the flag description/notes in the file.

    Returns:
        DataFrame with columns 'well' and 'notes', deduplicated and sorted.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File {path} not found")
    if path.endswith(".xlsx"):
        flagged = pd.read_excel(path)
    elif path.endswith(".csv"):
        flagged = pd.read_csv(path)
    else:
        raise ValueError(f"File {path} is not an Excel or CSV file.")
    flagged.rename(columns={well_col: "well", desc_well: "notes"}, inplace=True)
    flagged["well"] = flagged["well"].astype(str).str.strip().str.upper()
    flagged = flagged[flagged["well"].ne("") & flagged["well"].notna()]
    flagged = flagged.drop_duplicates(subset=["well"])
    flagged = flagged.sort_values(by=["well"]).reset_index(drop=True)
    return flagged


def add_flag_column(
    measurements: pd.DataFrame,
    flagged_wells: pd.DataFrame | str,
    well_col: str = "well",
    desc_well: str = "notes",
) -> pd.DataFrame:
    """Add an 'is_flagged' boolean column marking problematic wells.

    Args:
        measurements: DataFrame with a 'well' column.
        flagged_wells: DataFrame or path to file containing wells to flag.
        well_col: Column name for well identifiers in flagged_wells.
        desc_well: Column name for notes in flagged_wells.

    Returns:
        Copy of measurements with an added 'is_flagged' boolean column.
    """
    if isinstance(flagged_wells, pd.DataFrame):
        flags = flagged_wells.copy()
        if well_col != "well" or desc_well != "notes":
            flags = flags.rename(columns={well_col: "well", desc_well: "notes"})
        flags["well"] = flags["well"].astype(str).str.strip().str.upper()
        flags = flags[flags["well"].ne("") & flags["well"].notna()]
        flags = flags.drop_duplicates(subset=["well"]).sort_values(by=["well"]).reset_index(drop=True)
    elif isinstance(flagged_wells, str):
        flags = read_flagged_wells(flagged_wells, well_col, desc_well)
    else:
        raise ValueError("Flagged wells must be a DataFrame or a path")
    measurements = measurements.copy()
    mask = measurements["well"].isin(flags["well"])
    measurements.insert(len(measurements.columns), "is_flagged", mask)
    return measurements


def drop_flags(measurements: pd.DataFrame, flags: str | pd.DataFrame | None = None) -> pd.DataFrame:
    """Remove flagged wells from a measurements table.

    Args:
        measurements: DataFrame with a 'well' column and optionally an
            'is_flagged' boolean column.
        flags: Wells to remove, as a file path (str), DataFrame with a 'well'
            column, or list of well names. If None, removes rows where
            'is_flagged' is True.

    Returns:
        Copy of measurements with flagged wells removed.
    """
    if flags is None and "is_flagged" in measurements.columns:
        return measurements[measurements["is_flagged"] == False]
    if isinstance(flags, str):
        flags = read_flagged_wells(flags)
    if isinstance(flags, pd.DataFrame):
        return measurements[~measurements["well"].isin(flags["well"])]
    if isinstance(flags, list):
        flags = sorted(set(w.upper() for w in flags))
        return measurements[~measurements["well"].isin(flags)]


def drop_col(df: pd.DataFrame, col_num: int) -> pd.DataFrame:
    """Return a copy of df with all wells from the given plate column removed."""
    return df[df["well"].str[1:].astype(int) != col_num]


def drop_row(df: pd.DataFrame, row_letter: str) -> pd.DataFrame:
    """Return a copy of df with all wells from the given plate row removed."""
    return df[df["well"].str[0] != row_letter.strip().upper()]


def drop_well(df: pd.DataFrame, w: str) -> pd.DataFrame:
    """Return a copy of df with the specified well removed."""
    return df[df["well"] != w]

import pandas as pd
import datetime as dt
import numpy as np
import warnings
from pathlib import Path


def _normalize_time_to_timedelta(time_col: pd.Series) -> pd.Series:
    s = time_col.copy()
    is_time = s.map(lambda v: isinstance(v, dt.time))
    if is_time.any():
        s.loc[is_time] = s.loc[is_time].astype(str)
    td = pd.to_timedelta(s, errors="coerce")
    if td.isna().any():
        bad = time_col[td.isna()].tolist()
        raise ValueError(f"Unparseable Time values: {bad}")
    return td


def convert_excel_col_to_index(col: str | int) -> int:
    """Convert Excel column letters to a 0-based index (A->0, Z->25, AA->26)."""
    if isinstance(col, int):
        if col < 0:
            raise ValueError(f"Invalid Excel column: {col!r}")
        warnings.warn("Integer column indices are assumed to be 0-based.")
        return col
    s = col.strip().upper()
    if not s or any(not ("A" <= ch <= "Z") for ch in s):
        raise ValueError(f"Invalid Excel column: {col!r}")
    idx = 0
    for ch in s:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _read_gen5_wide_kinetics_table(
    path_to_data: str | Path, header_row: int, last_row: int | None, start_col: str | int
) -> pd.DataFrame:
    if not Path(path_to_data).is_file():
        raise FileNotFoundError(f"File {path_to_data} does not exist")
    if not path_to_data.endswith(".xlsx") and not path_to_data.endswith(".csv"):
        suffix = Path(path_to_data).suffix.lower()
        raise ValueError(f"Unsupported file type: {suffix}. Expected .xlsx or .csv.")
    header_idx = header_row - 1
    if last_row is None:
        if path_to_data.endswith(".xlsx"):
            df = pd.read_excel(path_to_data, header=header_idx)
        else:
            df = pd.read_csv(path_to_data, header=header_idx)
    else:
        n_rows = last_row - header_row
        if path_to_data.endswith(".xlsx"):
            df = pd.read_excel(path_to_data, header=header_idx, nrows=n_rows)
        else:
            df = pd.read_csv(path_to_data, header=header_idx, nrows=n_rows)
    if start_col != 0:
        start_col = convert_excel_col_to_index(start_col)
        df = df.iloc[:, start_col:].copy()
    if "Time" not in df.columns:
        raise KeyError(
            "Required column 'Time' not found. "
            "Check header_row, start_col, or the input file format."
        )
    df["Time"] = _normalize_time_to_timedelta(df["Time"])
    return df


def _convert_wide_to_tidy(data: pd.DataFrame, timepoint_cols: list) -> pd.DataFrame:
    for col in timepoint_cols:
        if col not in data.columns:
            raise ValueError(f"Column {col} does not exist.")
    value_cols = [col for col in data.columns if col not in timepoint_cols]
    tidy = data.melt(
        id_vars=timepoint_cols,
        value_vars=value_cols,
        var_name="well",
        value_name="od",
    )
    tidy = tidy.sort_values(by=["Time", "well"])
    tidy.reset_index(drop=True, inplace=True)
    return tidy


def _normalize_column_names_gen5_wide(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    for col in data.columns:
        if col == "Time":
            df = df.rename(columns={col: "time"})
        elif col == "T° 600":
            df = df.rename(columns={col: "temp_c"})
    return df


def _add_time_hours_from_timedelta(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["time"] = pd.to_timedelta(data["time"])
    data["time_hours"] = data["time"].dt.total_seconds() / 3600.0
    return data


def read_plate_measurements(
    reader_model: str,
    data_format: str,
    timepoint_cols: list | tuple,
    path: str,
    header_row: int = 1,
    last_row: int | None = None,
    start_col: str = 0,
) -> pd.DataFrame:
    """Read plate reader output and return a tidy DataFrame.

    Args:
        reader_model: Plate reader model (currently supported: "Synergy H1").
        data_format: Export format (currently supported: "wide").
        timepoint_cols: Columns that apply across the full plate at each timepoint.
        path: Path to the data file (.xlsx or .csv).
        header_row: 1-based row number of the column header.
        last_row: Last data row to read (inclusive). None reads to end of file.
        start_col: First data column as an Excel letter (e.g. "B") or 0-based int.

    Returns:
        Tidy DataFrame with columns: time, time_hours, well, od, plus any
        shared timepoint columns.
    """
    if timepoint_cols is None or len(timepoint_cols) == 0:
        raise ValueError("timepoint_cols must be provided and contain at least one column name.")
    if isinstance(timepoint_cols, str):
        raise ValueError("timepoint_cols must be a list or tuple of column names, not a single string.")
    if reader_model == "Synergy H1":
        if data_format == "wide":
            df = _convert_wide_to_tidy(
                _read_gen5_wide_kinetics_table(path, header_row, last_row, start_col),
                timepoint_cols,
            )
            df = _normalize_column_names_gen5_wide(df)
            df = _add_time_hours_from_timedelta(df)
            return df
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    else:
        raise ValueError(f"Unsupported reader model: {reader_model}")


def _read_plate_layout_column_blocks(path: str) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        raw = path
    else:
        raw = pd.read_excel(path)
    cols = raw.columns[1:]
    col_nums = raw.iloc[0, 1:].to_numpy().astype(int)
    clean_cond = np.array([c.split(".")[0] for c in cols])
    cond_names = list(dict.fromkeys(clean_cond))
    plate_cols = list(dict.fromkeys(col_nums))
    df = raw.iloc[1:, :].copy()
    df.set_index(df.columns[0], inplace=True)
    design = pd.DataFrame(columns=["well"] + cond_names)
    for row_label, row in df.iterrows():
        for col_num in plate_cols:
            well_name = f"{row_label}{col_num}"
            values = []
            for cond in cond_names:
                mask = (clean_cond == cond) & (col_nums == col_num)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    raise KeyError(f"No column found for condition '{cond}' at plate column {col_num}")
                j = indices[0]
                col_name = cols[j]
                values.append(row[col_name])
            design.loc[len(design)] = [well_name] + values
    return design


def read_plate_layout(path: str, format: str | None = None, data_format: str | None = None) -> pd.DataFrame:
    """Read a plate layout file and return a tidy per-well design table.

    Args:
        path: Path to the layout file or a DataFrame (for testing).
        format: Layout format. Currently supported: "column_blocks".
        data_format: Alias for ``format`` (backwards compatibility).

    Returns:
        Tidy DataFrame with one row per well and one column per condition.
    """
    fmt = format if format is not None else data_format
    if fmt == "column_blocks":
        return _read_plate_layout_column_blocks(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def merge_with_layout(
    measurements: pd.DataFrame, layout: pd.DataFrame
) -> pd.DataFrame:
    """Left-join measurements with a plate layout table on the 'well' column."""
    return measurements.merge(layout, on="well", how="left")

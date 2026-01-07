import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import linregress, t
from pathlib import Path
import warnings

def _normalize_time_to_timedelta(time_col:pd.Series) -> pd.Series:
    """Normalizes a time column into pandas Timedelta format.

       Args:
           time_col (pd.Series): The time column to normalize.

       Returns:
           pd.Database: The normalized time column in a pandas Timedelta format.

       Raises:
           ValueError: If the Time values cannot be normalized. """
    s = time_col.copy()

    # Only stringify the one type that hard-crashes
    is_time = s.map(lambda v: isinstance(v, dt.time))
    if is_time.any(): # If any time points are a datetime.time type
        s.loc[is_time] = s.loc[is_time].astype(str)  # stringify them so they can turn into pd.timedelta later

    td = pd.to_timedelta(s, errors="coerce") # convert everyone to timedelta

    if td.isna().any(): # If any time value in the series wasn't converted to timedelta
        bad = time_col[td.isna()].tolist() # here are the bad values
        raise ValueError(f"Unparseable Time values: {bad}") # raise an error to show them
    return td

def convert_excel_col_to_index(col: str | int) -> int:
    """Convert Excel column letters to a 0-based index.
         A->0, Z->25, AA->26, AB->27
         Args:
             col (str): Excel column letter to convert.

           Returns:
               int: The 0-based index of the column.

           Raises:
               ValueError: If the column is invalid. """
    if isinstance(col, int):
        if col < 0: raise ValueError(f"Invalid Excel column: {col!r}")
        warnings.warn("Integer column indices are assumed to be 0-based.")
        return col
    s = col.strip().upper() # Remove spaces and ensure upper case
    if not s or any(not ("A" <= ch <= "Z") for ch in s):
        raise ValueError(f"Invalid Excel column: {col!r}")
    idx = 0 # index to be returned
    for ch in s: # loop through the string and
        idx = idx * 26 + (ord(ch) - ord("A") + 1) # Excel columns work like base-26 numbers
    return idx - 1


def _read_gen5_wide_kinetics_table(path_to_data: str | Path, header_row, last_row: int | None, start_col:str|int) -> pd.DataFrame:
    """Accepts data directly from the plate reader and returns it in a wide format df
         Args:
             path_to_data (str | Path): path to data file
             header_row (int): The row index of the header
             last_row (int): The last row of the data file
             start_col (str): The start column of the data file

           Returns:
               pd.DataFrame: The wide data frame

           Raises:
               FileNotFoundError: If the data file does not exist.
               ValueError: If the data file cannot be read.
               KeyError: If the data file does not contain a column. """
    if not Path(path_to_data).is_file():
        raise FileNotFoundError(f"File {path_to_data} does not exist")
    if not path_to_data.endswith(".xlsx") and not path_to_data.endswith(".csv"):
        suffix = Path(path_to_data).suffix.lower()
        raise ValueError(f"Unsupported file type: {suffix}. Expected .xlsx or .csv.")
    header_idx = header_row -1

    if last_row is None:
        if path_to_data.endswith(".xlsx"):
            df = pd.read_excel(path_to_data, header = header_idx)
        else:
            df = pd.read_csv(path_to_data, header = header_idx)
    else:
        n_rows = last_row - header_row
        if path_to_data.endswith(".xlsx"):
            df = pd.read_excel(path_to_data, header = header_idx, nrows = n_rows)
        else:
            df = pd.read_csv(path_to_data, header = header_idx, nrows = n_rows)

    if start_col !=0: # Optional - skip columns
        start_col = convert_excel_col_to_index(start_col)
        df = df.iloc[:, start_col:].copy()

    if "Time" not in df.columns: # Make sure you have a time column
        raise KeyError("Required column 'Time' not found. "
            "Check header_row, start_col, or the input file format.")
    # Normalize format and data types
    df["Time"] = _normalize_time_to_timedelta(df["Time"])
    #df = df.drop(columns=["Time"])
    return df

def _convert_wide_to_tidy(data:pd.DataFrame, timepoint_cols) -> pd.DataFrame:
    """ Takes in a wide format dataframe and returns it in a tidy format
         Args:
             data (pd.DataFrame): The wide data frame
             timepoint_cols (list[str]): columns that apply across the plate at each time point

           Returns:
               pd.DataFrame: The wide data frame

           Raises:
               ValueError: If the data file does not contain a column mentioned in timepoint_cols. """
    for col in timepoint_cols:
        if col not in data.columns:
            raise ValueError(f"Column {col} does not exist.")
    value_cols = [col for col in data.columns if col not in timepoint_cols]
    tidy = data.melt(
        id_vars=timepoint_cols,
        value_vars=value_cols,
        var_name="well",
        value_name="od"
    )
    tidy = tidy.sort_values(by=["Time", "well"])
    tidy.reset_index(drop=True, inplace=True)
    return tidy

def _normalize_column_names_gen5_wide(data:pd.DataFrame) -> pd.DataFrame:
    """ Normalizes column names as found in a Gen5 wide kinetics table"""
    df = data.copy()
    for col in data.columns:
        if col == "Time":
            df = df.rename(columns={col: "time"})
        elif col == "T° 600":
            df = df.rename(columns={col: "temp_c"})
    return df

def _add_time_hours_from_timedelta(data:pd.DataFrame) -> pd.DataFrame:
    """ Given a DataFrame with a time column that is already a pandas Timedelta (or convertible to one), add a numeric time_hours column.
        Args:
             data (pd.DataFrame): tidy format data frame with a time column that is a pandas Timedelta
        Returns:
            pd.DataFrame: tidy format data frame with an added time column in hours elapsed.  ”"""
    data = data.copy()
    data["time"] = pd.to_timedelta(data["time"])
    data["time_hours"] = data["time"].dt.total_seconds() / 3600.0

    return data

def read_plate_measurements(reader_model:str, data_format:str, timepoint_cols:list|tuple, path:str, header_row:int=1, last_row:int|None=None, start_col:str=0) -> pd.DataFrame:
    """ Reads the measurements from the plate and returns a tidy format data frame.
    Args:
        reader_model (str): The reader model
        data_format (str): The data format
        timepoint_cols (list[str]): columns that apply across the plate at each time point
        path (str): The path to the data file
        header_row (int): The row index of the header
        last_row (int): The last row of the data file
        start_col (str): The start column of the data file
    Returns:
        pd.DataFrame: The tidy format data frame
    Raises:
        ValueError: If the data format or reader model are not supported.  """
    if timepoint_cols is None or len(timepoint_cols) ==0:
        raise ValueError("timepoint_cols must be provided and contain at least one column name.")
    if isinstance(timepoint_cols, str):
        raise ValueError("timepoint_cols must be a list or tuple of column names, not a single string.")
    if reader_model == "Synergy H1":
        if data_format == "wide":
            df= _convert_wide_to_tidy(_read_gen5_wide_kinetics_table(path, header_row, last_row, start_col), timepoint_cols)
            df = _normalize_column_names_gen5_wide(df)
            df = _add_time_hours_from_timedelta(df)
            return df
        else:
            raise ValueError(f"Unsupported data format: {reader_model}")
    else:
        raise ValueError(f"Unsupported reader model: {reader_model}")

def _read_plate_layout_column_blocks(path:str) -> pd.DataFrame:
    """ Read a column-block plate layout and return a tidy per-well design table.

      The input must be a grid-style layout where:
      - The first column contains plate row labels (A, B, C, ...).
      - Condition columns are repeated per plate column (e.g. strain.1, strain.2).
      - The first data row maps each condition block to a plate column number (1–12).

      Args:
          path: Path to an Excel file or a DataFrame containing the plate layout.

      Returns:
          A tidy DataFrame with one row per well (e.g. A1, A2, ...) and one column
          per condition. """
    if isinstance(path, pd.DataFrame):     # Load data. Optional only for testing
        raw = path
    else:
        raw = pd.read_excel(path)

    # Identify the columns: raw.columns[0] is the row label column (A, B, C, ...), raw.columns[1:] are the condition columns like strain.1, strain.2, bio_rep.1
    cols = raw.columns[1:]
    col_nums = raw.iloc[0, 1:].to_numpy().astype(int) # actual plate column numbers for each metadata column (numbers to construct A1 A2 A3)
    clean_cond = np.array([c.split(".")[0] for c in cols])  # Condition base names: strip pandas' ".1",".2" etc
    cond_names = list(dict.fromkeys(clean_cond))     # Unique condition names in original order (dict.fromkeys preserves insertion order)

    plate_cols = list(dict.fromkeys(col_nums))     # Unique plate column numbers in the order they appear
    df = raw.iloc[1:, :].copy() # drop the value_type row
    df.set_index(df.columns[0], inplace=True)  # index is now the letters (A , B )

    design = pd.DataFrame(columns=["well"] + cond_names) # Empty dataframe for results, well is the only known column currently
    for row_label, row in df.iterrows(): # For each row (A-H), row is a series with entries for strain.1, strain.2 etc
        for col_num in plate_cols: # For each column (strain.1, strain.2 still exist)
            well_name = f"{row_label}{col_num}"
            values = [] # We'll collect the condition values for this well
            # For each condition type
            for cond in cond_names:
                # Find which raw column stores this condition for this plate column number
                # Example: cond="strain" and col_num=2 should match "strain.2"
                mask = (clean_cond == cond) & (col_nums == col_num)
                indices = np.where(mask)[0]
                if len(indices) == 0: # If we can't find a matching column, the design file is inconsistent
                    raise KeyError(f"No column found for condition '{cond}' at plate column {col_num}")
                j = indices[0]     # index into cols / clean_cond / col_nums
                col_name = cols[j]    # actual column name in df, e.g. "strain.1" or "strain.3"
                values.append(row[col_name])

            design.loc[len(design)] = [well_name] + values
    return design

def read_plate_layout(path:str, format:str):
    """Read a plate layout file and return a tidy per-well design table.

     Args:
         path: Path to a plate layout file or a DataFrame.
         data: Plate layout format. Currently supported: "column_blocks".

     Returns:
         A tidy DataFrame with one row per well and one column per condition.

     Raises:
         ValueError: If the requested layout format is not supported. """
    if format == "column_blocks":
        return _read_plate_layout_column_blocks(path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def merge_measurements_and_conditions(measurements:pd.DataFrame, conditions:pd.DataFrame)-> pd.DataFrame:
    """ Take in a measurement data frame (tidy) and a design data frame and merge them """
    full = measurements.merge(conditions, on="well", how="left")
    return full

def drop_col(df:pd.DataFrame, col_num:int) -> pd.DataFrame:
    """ Takes in a DataFrame and a column. Returns a copy without the target column"""
    df = df[df["well"].str[1:].astype(int) != col_num]
    return df

def drop_row(df:pd.DataFrame, row_letter:str) -> pd.DataFrame:
    """ Takes in a DataFrame and a row. Returns a copy without the target row"""
    row_letter = row_letter.strip().upper()
    df = df[df["well"].str[0] != row_letter]
    return df

def drop_well(df:pd.DataFrame, w:str) -> pd.DataFrame:
    """ Takes in a DataFrame and a well. Returns a copy without the target well"""
    df = df[df["well"] != w]
    return df

def read_flagged_wells(path: str, well_col: str = "well", desc_well:str= "notes") -> pd.DataFrame:
    """ Reads a table of flagged wells with their reasoning and returns it
        Args:
            path (str): Path to the data file
            well_col (str): The name of the well column in the data file
            desc_well (str): The name of the well description column in the data file
        Returns:
            pd.DataFrame: The tidy format data frame
        Raises:
            FileNotFoundError: If the flags file does not exist.
            ValueError: If the data format is not supported. """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File {path} not found")
    if path.endswith(".xlsx"):
        flagged = pd.read_excel(path)
    elif path.endswith(".csv"):
        flagged = pd.read_csv(path)
    else:
        raise ValueError(f"File {path} is not an Excel file.")
    flagged.rename(columns={well_col: "well", desc_well:"notes"}, inplace=True)
    # normalize wells like "a1" -> "A1"
    flagged["well"] = flagged["well"].astype(str).str.strip().str.upper()
    # drop blanks + duplicates
    flagged = flagged[flagged["well"].ne("") & flagged["well"].notna()]
    flagged = flagged.drop_duplicates(subset=["well"])
    flagged = flagged.sort_values(by=["well"]).reset_index(drop=True)
    return flagged

# Create a good docking for the add_flag_column function

def add_flag_column(measurements: pd.DataFrame, flagged_wells: pd.DataFrame|str, well_col: str = "well", desc_well:str="notes") -> pd.DataFrame:
    """
    Adds a boolean column to measurements based on whether 'well'
    appears in flagged_wells['well'].
         Args:
            measurements (pd.DataFrame): DataFrame of measurements with a 'well' column.
            flagged_wells (pd.DataFrame | str): DataFrame or path to file containing flagged wells.
            well_col (str): Name of the well column in flagged_wells
            desc_well (str): Name of the description column in flagged_wells
         Returns:
            pd.DataFrame: Copy of measurements with an added 'is_flagged' boolean column.
    
        """
    if isinstance(flagged_wells, pd.DataFrame):
        flags = flagged_wells.copy()
        # Normalize column names and values to match internal convention
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
    measurements.insert(len(measurements.columns), 'is_flagged', mask)
    return measurements

def drop_flags(measurements:pd.DataFrame, flags:str|pd.DataFrame|None=None)-> pd.DataFrame:
    """Remove flagged wells from a measurements table.
    If ``flags`` is provided it can be one of:
    - a path to an Excel/CSV file readable by :func:`read_flagged_wells` (string),
    - a DataFrame containing a ``well`` column,
    - a list of well names (strings).

    Args:
        measurements (pd.DataFrame): DataFrame of measurements with a ``well``
            column and optionally an ``is_flagged`` boolean column.
        flags (str | pd.DataFrame | list | None): Flags specification as
            described above. If a path or DataFrame is provided, wells listed
            in that source will be removed. If a list is provided it is
            treated as case-insensitive well names to drop.

    Returns:
        pd.DataFrame: A copy of ``measurements`` with flagged wells removed.

    Raises:
        FileNotFoundError: If a provided file path does not exist (raised by
            :func:`read_flagged_wells`).
    """
    # if no flags is provided, it's assumed that measurements already contains flags.
    # If flags are provided, they override existing flags
    if flags is None and "is_flagged" in measurements.columns:
        return measurements[measurements["is_flagged"] == False]
    # Only getting here if flags were provided (at this point, it doesn't matter if measurement contains flags or not)
    if isinstance(flags, str): # If flags wasn't a dataframe, make it one
        flags = read_flagged_wells(flags)
    if isinstance(flags, pd.DataFrame):
        # Keep only wells NOT present in the flags table (exclude flagged wells)
        return measurements[~measurements["well"].isin(flags["well"])]
    if isinstance(flags, list):
        # Normalize and exclude wells listed in the provided list
        flags = [i.upper() for i in flags]
        flags = list(set(flags))
        flags.sort()
        return measurements[~measurements["well"].isin(flags)]

def with_blank_corrected_od(df:pd.DataFrame, window:int=4, od_col:str="od") -> pd.DataFrame:
    """Add blank-corrected OD column.

    Args:
        df (pd.DataFrame): Measurements with 'well' and time columns.
        window (int): Number of initial points to average.
        od_col (str): Name of the OD column to use.

    Returns:
        pd.DataFrame: Copy with an added ``od_blank`` column.
    """
    df = df.sort_values(["well", "time_hours"]).copy()
    df["od_blank"] = np.nan

    for well, group in df.groupby("well", sort=False): # For each well
        blank_value = group[od_col].iloc[:window].mean()         #Take the first `window` measurements in this wel, compute their average
        blanked = (group[od_col] - blank_value).clip(lower=0)         # subtract blank from every value. if lower than 0 , replace with 0
        df.loc[group.index, "od_blank"] = blanked.to_numpy()         # Write back using the original indices
    df = df.sort_values(["time", "well"]).reset_index(drop=True)
    return df

def with_smoothed_od(df:pd.DataFrame, group_by: str|list[str]= "well", od:str = "od_blank",  window:int=5) -> pd.DataFrame:
    """Return a copy with an added ``od_smooth`` column.

    Args:
        df (pd.DataFrame): Measurements containing a ``well`` and time column.
        group_by (str | list): Column(s) to group by when smoothing.
        od (str): Name of the OD column to smooth (default: ``od_blank``).
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: Copy with an ``od_smooth`` column added.
    """
    if isinstance(group_by, list):
        sort_by = ["time_hours"] + group_by
    else:
        sort_by =  ["time_hours"] + [group_by]
    df = df.copy()
    df["od_smooth"] = np.nan # Create a new column for the smooth data
    df = df.sort_values(by=sort_by)
    for well, group in df.groupby(group_by):  # well = name (int), group = df for that well
        smoothed = (group[od].rolling(window, center=True, min_periods=1).mean()) # Replace value with the mean of window cells around it
        df.loc[group.index, "od_smooth"] = smoothed
    df.sort_values(by=sort_by, inplace=True)
    return df


def _calc_growth_rate(x:pd.Series, y:pd.Series, window:int, epsilon:float)-> np.array:
    """
    Args:
        x (pd.Series): time series
        y (pd.Series): OD series
        window (int): window size for regression line
        epsilon (float): what value is too small to be considered as biologically significance / actual
    Return:
        an array of the growth rates per time point
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    logy = np.log(np.where(y > epsilon, y, np.nan)) # Set value to NaN where OD is lower than epsilon so log is not taken, and take the log
    n = len(x)
    growth_rate = np.full(n, np.nan)
    half = window // 2  # how many points on each side of i

    for i in range(n):
        start = max(0, i - half)
        end   = min(n, i + half + 1)   # slice is [start, end)

        xs = x[start:end]
        ys = logy[start:end]
        valid = np.isfinite(ys) # drop NaNs in ys

        if valid.sum() < 2:
            continue    # not enough points to fit a line

        xs_win = xs[valid]
        ys_win = ys[valid]

        slope, intercept = np.polyfit(xs_win, ys_win, 1) # fit a line

        growth_rate[i] = slope         # slope is the growth rate at x[i]
    return growth_rate

def add_growth_rate(df:pd.DataFrame, window:int=5, epsilon:float= 1e-10, group_by: str|list[str] = "well", od:str = "od_smooth") -> pd.DataFrame:
    """Add a per-time-point growth rate column ('mu').

    Args:
        df (pd.DataFrame): Measurements containing a time column and grouping column(s).
        window (int): Number of points used for the local regression window.
        epsilon (float): Minimum OD value treated as valid for log transform.
        group_by (str | list[str]): Column name(s) to group by when computing rates.
        od (str): Name of the OD column to use for regression (default: ``od_smooth``).

    Returns:
        pd.DataFrame: Copy of ``df`` with an added ``mu`` column containing growth rates.
    """
    if isinstance(group_by, list):
        sort_by = ["time_hours"] + group_by
    else:
        sort_by =  ["time_hours"] + [group_by]
    df = df.copy()
    df["mu"] = np.nan
    # For each unique well, group them
    for well, group in df.groupby(group_by):
        group = group.sort_values("time_hours")
        d_values = _calc_growth_rate(group["time_hours"], group[od], window,epsilon)
        df.loc[group.index, "mu"] = d_values

    df.sort_values(by=sort_by, inplace=True)
    return df

def estimate_early_od_threshold(df:pd.DataFrame, od_col:str="od_smooth", n_points:int=4, q:float=0.95)-> float:
    """Estimate an early-OD threshold from initial timepoints.

    Args:
        df (pd.DataFrame): Measurements containing 'well' and time columns.
        od_col (str): Name of the OD column to use.
        n_points (int): Number of initial points per well to consider.
        q (float): Quantile; OD above this percentile is considered valid.

    Returns:
        float: Estimated OD threshold.
    """
    # subset: first n_points per well
    early = (df.sort_values(["well", "time_hours"]).groupby("well").head(n_points))
    od_low = early[od_col].quantile(q)
    return od_low

def _calc_mu_max(x, y, w, threshold, epsilon=1e-10):
    """
    Find the maximum growth rate over sliding windows.

    Fits linear regressions on log-transformed OD values across sliding
    windows of width ``w`` and returns the largest slope (mu) found along
    with a simple approximate 95% confidence interval using the reported
    standard error from the best-fit window.

    Args:
        x (array-like): Time points.
        y (array-like): OD measurements corresponding to ``x``.
        w (int): Window size (number of consecutive points) to scan.
        threshold (float): Minimum OD cutoff to consider a window valid.
        epsilon (float): Small positive value used as a lower bound for the
            cutoff to avoid log(0).

    Returns:
        tuple: (best_mu, mu_low, mu_high) where each is a float or ``np.nan``
            when no valid window was found or the CI could not be computed.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    n = len(x)
    best_mu = -np.inf
    std = np.nan
    for i in range(n):
        x_slice = x[i:i+w]
        y_slice = y[i:i+w]
        valid = np.isfinite(y_slice) # drop NaNs in ys
        cutoff = max(threshold, epsilon)

        if valid.sum() < w or not np.all(y_slice > cutoff):
            continue    # not enough points to fit a line
        logy = np.log(y_slice)
        res = linregress(x_slice, logy)
        mu = float(res.slope)
        if mu >best_mu:
            best_mu = mu
            std = float(res.stderr)

    # Done looking at all windows, compute the actual values now
    if best_mu == -np.inf: # If best_mu is still -infinity, no one replaced it so there's no mu max (no one qualified)
        return np.nan, np.nan, np.nan
    d_free = w - 2
    if d_free <= 0 or not np.isfinite(std): return best_mu, np.nan, np.nan # Do I have enough information to compute a valid confidence interval for mu

    t_crit = float(t.ppf(0.975, d_free))
    mu_low = best_mu - t_crit * std
    mu_high = best_mu + t_crit * std

    return best_mu, mu_low, mu_high

def summarize_mu_max(df, group_by = "well", window=5, od = "od_smooth", threshold=None):
    """Estimate mu_max (max growth rate) per group.

    Scans each group's OD time series with a sliding regression window to
    find the maximum exponential growth rate (mu). Also converts the
    reported slope into doubling time (tau) and returns simple 95%
    confidence-interval endpoints for mu when available.

    Args:
        df (pd.DataFrame): Measurements containing a ``well`` (or grouping)
            column and ``time_hours``.
        group_by (str | list): Column(s) to group by when computing mu_max.
        window (int): Window size (number of points) used for local fits.
        od (str): Column name of the OD values to use.
        threshold (float | None): If None, estimated from early timepoints;
            otherwise used as the minimum OD cutoff for valid windows.

    Returns:
        pd.DataFrame: Table with columns ``['well','mu_max','mu_low',
        'mu_high','tau','tau_low','tau_high']`` containing the results per
        group.
    """
    #group_by = group_by if isinstance(group_by, list) else [group_by]
    if threshold is None:
        threshold = estimate_early_od_threshold(df, od_col=od, n_points=window, q=0.95)
    result = pd.DataFrame(columns=['well', 'mu_max', 'mu_low', 'mu_high','tau', 'tau_low', 'tau_high'  ])
    for key, group in df.groupby(group_by): # For each well/group to calc mu_max for
        group = group.sort_values("time_hours")
        best_mu, mu_low, mu_high = _calc_mu_max(group["time_hours"], group[od], window, threshold)
        # If no valid window was found, _calc_mu_max returns NaNs
        if np.isnan(best_mu) and np.isnan(mu_low) and np.isnan(mu_high):
            print(f"No meaningful growth found for this group: {key}")
        tau = np.log(2) / best_mu if best_mu > 0 else np.nan
        tau_2p5 = np.log(2) / mu_high if mu_high > 0 else np.nan  # fastest
        tau_97p5 = np.log(2) / mu_low if mu_low > 0 else np.nan  # slowest

        result.loc[len(result)] = [key, best_mu, mu_low, mu_high, tau, tau_2p5, tau_97p5]
    return result

# Backwards-compatible aliases for older API names (module-level)
add_blank_value = with_blank_corrected_od
add_smoothed_od = with_smoothed_od
mu_max_create = summarize_mu_max
read_plate_design = read_plate_layout




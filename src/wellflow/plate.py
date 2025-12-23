import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import linregress, t

def _normalize_time(time_col:pd.Series) -> pd.Series:
    """
     Takes in a time series and normalizes its type.
    """
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
    """
    Convert Excel column letters to a 0-based index.
     A->0, Z->25, AA->26, AB->27
    """
    if isinstance(col, int):
        return col
    s = col.strip().upper() # Remove spaces and ensure upper case
    if not s or any(not ("A" <= ch <= "Z") for ch in s):
        raise ValueError(f"Invalid Excel column: {col!r}")
    idx = 0 # index to be returned
    for ch in s: # loop through the string and
        idx = idx * 26 + (ord(ch) - ord("A") + 1) # Excel columns work like base-26 numbers
    return idx - 1


def _read_wide_table(path:str, header_row:int|None, last_row:int|None, start_col:str|None) -> pd.DataFrame:
    """
    Accepts data directly from the plate reader and returns it in a wide format df
    """
    if header_row is None:
        header_row = 1
        header_idx = 0
    else:
        header_idx = header_row -1
    if last_row is None:
        df = pd.read_excel(path, header = header_idx)
    else:
        n_rows = last_row - header_row
        df = pd.read_excel(path, header = header_idx, nrows = n_rows)

    if start_col is not None: # Optional - skip columns
        start_col = convert_excel_col_to_index(start_col)
        df = df.iloc[:, start_col:].copy()

    if "Time" not in df.columns: # Make sure you have a time column
        raise KeyError("Required column 'Time' not found. "
            "Check header_row, start_col, or the input file format.")
    # Normalize format and data types
    df["time"] = _normalize_time(df["Time"])
    df = df.drop(columns=["Time"])
    if "T° 600" in df.columns:
        df = df.rename(columns={"T° 600": "temperature_c"})

    return df

def convert_wide_to_tidy(wide_df:pd.DataFrame, id_cols) -> pd.DataFrame:
    """
    Takes in a wide format dataframe and returns it in a tidy format
    id_cols are the columns that contain info per time point, not per well
    """
    value_cols = [col for col in wide_df.columns if col not in id_cols]
    tidy = wide_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="well",
        value_name="od"
    )
    tidy = tidy.sort_values(by=["time", "well"])
    tidy.reset_index(drop=True, inplace=True)
    return tidy

def _add_time_hours(df:pd.DataFrame) -> pd.DataFrame:
    """ Takes in the newly formed tidy dataframe and add a column of time in hours"""
    df = df.copy()
    df["time"] = pd.to_timedelta(df["time"])
    df["time_hours"] = df["time"].dt.total_seconds() / 3600.0

    return df

def read_plate_measurements(path:str, header_row:int=None, last_row:int=None, start_col:str=None, reader_model="Synergy H1", data_format="wide", id_cols=("time", "temperature_c")) -> pd.DataFrame:
    """
    Taking raw measurement data from the reader and returning it in a tidy format with time_hours
    """
    if reader_model == "Synergy H1":
        if data_format == "wide":
            df= convert_wide_to_tidy(_read_wide_table(path, header_row, last_row, start_col), id_cols)
            df = _add_time_hours(df)
            return df
    else:
        raise ValueError("Unknown reader model")

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

def parse_plate_design(path:str) -> pd.DataFrame:
    """
    Takes in a path to the design.
    Returns a tidy design table: one row per well (A1, A2, ...), columns are conditions
    """
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

def merge_measurements_and_conditions(measurements:pd.DataFrame, conditions:pd.DataFrame)-> pd.DataFrame:
    """
    Take in a measurement data frame (tidy) and a design data frame and merge them
    """
    full = measurements.merge(conditions, on="well", how="left")
    return full

def read_flagged_wells(path: str, well_col: str = "well", desc_well:str="notes") -> pd.DataFrame:
    """
    Reads an Excel file listing wells to flag
    Returns a DataFrame with a 'well' column and is_flagged column.
    """
    flagged = pd.read_excel(path)
    flagged = flagged.copy()
    flagged.rename(columns={well_col: "well", desc_well:"notes"}, inplace=True)
    # normalize wells like "a1" -> "A1"
    flagged["well"] = flagged["well"].astype(str).str.strip().str.upper()
    # drop blanks + duplicates
    flagged = flagged[flagged["well"].ne("") & flagged["well"].notna()]
    flagged = flagged.drop_duplicates(subset=["well"])
    flagged = flagged.sort_values(by=["well"]).reset_index(drop=True)
    return flagged

def add_flag_column(measurements: pd.DataFrame, flagged_wells: pd.DataFrame|str, well_col: str = "well", desc_well:str="notes") -> pd.DataFrame:
    """
    Adds a boolean column (default: is_flagged) to measurements based on whether 'well'
    appears in flagged_wells['well'].
    """
    if isinstance(flagged_wells, pd.DataFrame):
        flags = flagged_wells.copy()
    elif isinstance(flagged_wells, str):
        flags = read_flagged_wells(flagged_wells, well_col, desc_well)
    else:
        raise ValueError("Flagged wells must be a DataFrame or a path")
    mask = measurements["well"].isin(flags["well"])
    measurements.insert(len(measurements.columns), 'is_flagged', mask)
    return measurements

def drop_flags(measurements:pd.DataFrame, flags:str|pd.DataFrame|None=None)-> pd.DataFrame:
    """
    Can provide a list or a file of flags, and if you don't then it's the internal flags already stored in the measurements
    """
    # if no flags is provided, it's assumed that measurements already contains flags.
    # If flags are provided, they override existing flags
    if flags is None and "is_flagged" in measurements.columns:
        return measurements[measurements["is_flagged"] == False]
    # Only getting here if flags were provided (at this point, it doesn't matter if measurement contains flags or not)
    if isinstance(flags, str): # If flags wasn't a dataframe, make it one
        flags = read_flagged_wells(flags)
    if isinstance(flags, pd.DataFrame):
        return measurements[measurements["well"].isin(flags["well"])]
    if isinstance(flags, list):
        flags = [i.upper() for i in flags]
        flags = list(set(flags))
        flags.sort()
        return measurements[measurements["well"].isin(flags)]

def add_blank_value(df:pd.DataFrame, window:int=4, od_col:str="od") -> pd.DataFrame:
    """
    Takes in a dataframe and a window size and adds a blank value column.

    :param df: data frame of measurements you want to blank
    :param window: how many initial points should be used to construct the blank value
    :param od_col: name of the od column to use
    :return: returns the blanked data frame (copy)
    """
    df = df.sort_values(["well", "time_hours"]).copy()
    df["od_blank"] = np.nan

    for well, group in df.groupby("well", sort=False): # For each well
        blank_value = group[od_col].iloc[:window].mean()         #Take the first `window` measurements in this wel, compute their average
        blanked = (group[od_col] - blank_value).clip(lower=0)         # subtract blank from every value. if lower than 0 , replace with 0
        df.loc[group.index, "od_blank"] = blanked.to_numpy()         # Write back using the original indices
    df = df.sort_values(["time", "well"]).reset_index(drop=True)
    return df

def add_smoothed_od(df:pd.DataFrame, group_by: str|list[str]= "well", od:str = "od_blank",  window:int=5) -> pd.DataFrame:
    """

    :param df: Data frame of measurements you want to smooth
    :param group_by: defaults to smoothing by well but can smooth by a condition(s) for averaging functionality
    :param od: defaults to blank od but can be the mean of for averaging by condition
    :param window: the mean of how many points to average over
    :return: the smoothed data frame
    """
    if isinstance(group_by, list):
        sort_by = ["time_hours"] + group_by
    else:
        sort_by =  ["time_hours"] + [group_by]
    df = df.copy()
    df["od_smooth"] = np.nan # Create a new column for the smooth data

    for well, group in df.groupby(group_by):  # well = name (int), group = df for that well
        smoothed = (group[od].rolling(window, center=True, min_periods=1).mean()) # Replace value with the mean of window cells around it
        df.loc[group.index, "od_smooth"] = smoothed
    df.sort_values(by=sort_by, inplace=True)
    return df


def _calc_growth_rate(x:pd.Series, y:pd.Series, window:int, epsilon:float)-> np.array:
    """
        
    :param x: time series
    :param y: OD series
    :param window: window size for regression line
    :param epsilon: what value is too small to be considered as biologically significance / actual measurement
    :return: an array of the growth rates per time point
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
    """
    :param df: data frame to which you want to add growth rate        
    :param window: across how many points to calculate regression line
    :param epsilon: what value is too small to be considered as biologically significance / actual measurement
    :param group_by: defaults to calculating by well but can calculate by a condition(s) for averaging functionality
    :param od: defaults to od_smooth but can be the mean_od for averaging by condition
    :return: the data frame with added growth rate
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
    """
    :param df: data frame for which you want to calculate the early od threshold
    :param od_col: column name of OD series
    :param n_points: across how many of the initial points to calculate the early od threshold
    :param q: above this percentile, the OD is considered to be valid
    :return: the od threshold 
    """
    # subset: first n_points per well
    early = (df.sort_values(["well", "time_hours"]).groupby("well").head(n_points))
    od_low = early[od_col].quantile(q)
    return od_low

def _calc_mu_max(x, y, w, threshold, epsilon=1e-10):
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

def mu_max_create(df, group_by = "well", window=5,od="od_smooth", threshold=None):
    #group_by = group_by if isinstance(group_by, list) else [group_by]
    if threshold is None:
        threshold = estimate_early_od_threshold(df, od_col=od, n_points=window, q=0.95)
    result = pd.DataFrame(columns=['well', 'mu_max', 'mu_low', 'mu_high','tau', 'tau_low', 'tau_high'  ])
    for key, group in df.groupby(group_by): # For each well/group to calc mu_max for
        group = group.sort_values("time_hours")
        best_mu, mu_low, mu_high = _calc_mu_max(group["time_hours"], group[od], window, threshold)
        if best_mu is np.nan and mu_low is np.nan and mu_high is np.nan:
            print(f"No meaningful growth found for this group:{key}")
        tau = np.log(2) / best_mu if best_mu > 0 else np.nan
        tau_2p5 = np.log(2) / mu_high if mu_high > 0 else np.nan  # fastest
        tau_97p5 = np.log(2) / mu_low if mu_low > 0 else np.nan  # slowest

        result.loc[len(result)] = [key, best_mu, mu_low, mu_high, tau, tau_2p5, tau_97p5]
    return result




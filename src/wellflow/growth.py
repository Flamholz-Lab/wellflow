import pandas as pd
import numpy as np
from scipy.stats import linregress, t


def _calc_growth_rate(
    x: pd.Series, y: pd.Series, window: int, epsilon: float
) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    logy = np.log(np.where(y > epsilon, y, np.nan))
    n = len(x)
    growth_rate = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        xs = x[start:end]
        ys = logy[start:end]
        valid = np.isfinite(ys)
        if valid.sum() < 2:
            continue
        xs_win = xs[valid]
        ys_win = ys[valid]
        if xs_win.size < 2:
            continue
        slope, _ = np.polyfit(xs_win, ys_win, 1)
        growth_rate[i] = slope
    return growth_rate


def add_growth_rate(
    df: pd.DataFrame,
    window: int = 5,
    epsilon: float = 1e-10,
    group_by: str | list[str] = "well",
    od_col: str = "od_smooth",
) -> pd.DataFrame:
    """Add a per-timepoint growth rate column ('mu').

    Args:
        df: Measurements with 'time_hours' and grouping column(s).
        window: Number of points used for the local log-linear regression.
            Must be odd so the window is centered symmetrically around each point.
        epsilon: Minimum OD treated as valid for log transform.
        group_by: Column(s) to group by when computing rates.
        od_col: OD column to use for regression.

    Returns:
        Copy of df with an added 'mu' column (units: 1/hour).
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd (got {window}). "
                         f"Try {window - 1} or {window + 1}.")
    sort_by = ["time_hours"] + (group_by if isinstance(group_by, list) else [group_by])
    df = df.copy()
    df["mu"] = np.nan
    for _, group in df.groupby(group_by):
        group = group.sort_values("time_hours")
        df.loc[group.index, "mu"] = _calc_growth_rate(
            group["time_hours"], group[od_col], window, epsilon
        )
    df.sort_values(by=sort_by, inplace=True)
    return df


def estimate_od_threshold(
    df: pd.DataFrame, od_col: str = "od_smooth", n_points: int = 4, q: float = 0.95
) -> float:
    """Estimate a baseline OD threshold from the first few timepoints per well.

    Args:
        df: Measurements with 'well' and 'time_hours' columns.
        od_col: OD column to use. Works with raw, blank-corrected, or smoothed OD —
            the threshold adapts to whichever column you pass.
        n_points: Number of initial timepoints per well to consider.
        q: Quantile; OD above this value is treated as valid signal.

    Returns:
        OD threshold as a float.
    """
    early = df.sort_values(["well", "time_hours"]).groupby("well").head(n_points)
    return float(early[od_col].quantile(q))


def _calc_mu_max(
    x, y, w: int, threshold: float, epsilon: float = 1e-10
) -> tuple:
    """Find the maximum growth rate over sliding windows.

    Returns:
        (best_mu, mu_low, mu_high, t_center, t_start, t_end) — all NaN when
        no valid window was found.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    best_mu = -np.inf
    std = np.nan
    best_i = None
    cutoff = max(threshold, epsilon)
    for i in range(n):
        x_slice = x[i : i + w]
        y_slice = y[i : i + w]
        if not np.all(np.isfinite(y_slice)) or len(y_slice) < w or not np.all(y_slice > cutoff):
            continue
        logy = np.log(y_slice)
        res = linregress(x_slice, logy)
        mu = float(res.slope)
        if mu > best_mu:
            best_mu = mu
            std = float(res.stderr)
            best_i = i
    if best_mu == -np.inf:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    t_start = float(x[best_i])
    t_end = float(x[best_i + w - 1])
    t_center = 0.5 * (t_start + t_end)
    d_free = w - 2
    if d_free <= 0 or not np.isfinite(std):
        return best_mu, np.nan, np.nan, t_center, t_start, t_end
    t_crit = float(t.ppf(0.975, d_free))
    return best_mu, best_mu - t_crit * std, best_mu + t_crit * std, t_center, t_start, t_end


def compute_mu_max(
    df: pd.DataFrame,
    group_by: str | list[str] = "well",
    window: int = 5,
    od_col: str = "od_smooth",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Estimate mu_max (maximum growth rate) per group.

    Scans each group's OD time series with a sliding regression window.
    Also computes doubling time (tau) and 95% CI endpoints when available.

    Args:
        df: Measurements with 'time_hours' and grouping column(s).
        group_by: Column(s) to group by.
        window: Sliding window size (number of points).
        od_col: OD column to use.
        threshold: Minimum OD for a window to be considered. If None,
            estimated automatically from early timepoints.

    Returns:
        DataFrame with one row per group and columns:
        well, mu_max, mu_low, mu_high, tau, tau_low, tau_high,
        t_mu_max, t_start, t_end.
    """
    if threshold is None:
        threshold = estimate_od_threshold(df, od_col=od_col)
    result = pd.DataFrame(
        columns=["well", "mu_max", "mu_low", "mu_high", "tau", "tau_low", "tau_high", "t_mu_max", "t_start", "t_end"]
    )
    for key, group in df.groupby(group_by):
        group = group.sort_values("time_hours")
        best_mu, mu_low, mu_high, t_center, t_start, t_end = _calc_mu_max(
            group["time_hours"], group[od_col], window, threshold
        )
        if np.isnan(best_mu):
            print(f"No meaningful growth found for group: {key}")
        tau = np.log(2) / best_mu if best_mu > 0 else np.nan
        tau_low = np.log(2) / mu_high if (pd.notna(mu_high) and mu_high > 0) else np.nan
        tau_high = np.log(2) / mu_low if (pd.notna(mu_low) and mu_low > 0) else np.nan
        result.loc[len(result)] = [key, best_mu, mu_low, mu_high, tau, tau_low, tau_high, t_center, t_start, t_end]
    return result

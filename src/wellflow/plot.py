import os
import re
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_grouped(
    group_by,
    df,
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    plot_by=None,
    marker=None,
    log=False,
    font_size=14,
    xlim=None,
    ylim=None,
    save=False,
    palette=None,
):
    """Plot multiple series on one axes, colored by group.

    Args:
        group_by: Column name to color by.
        df: DataFrame containing the data.
        x_col: Column for the x-axis.
        y_col: Column for the y-axis.
        x_label: x-axis label.
        y_label: y-axis label.
        title: Plot title.
        plot_by: Column for individual lines within a group (disables aggregation).
        marker: Matplotlib marker style.
        log: If True, use a log scale on the y-axis.
        font_size: Base font size.
        xlim: (min, max) tuple for x-axis limits.
        ylim: (min, max) tuple for y-axis limits.
        save: If True, save the figure as a PNG next to where the script runs.
        palette: List of colors passed to seaborn.
    """
    required = {x_col, y_col, group_by}
    if plot_by is not None:
        required.add(plot_by)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Plot data is missing columns: {sorted(missing)}")

    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=group_by,
        palette=palette,
        units=plot_by if plot_by is not None else None,
        estimator=None if plot_by is not None else "mean",
        lw=1.2,
        alpha=0.7,
        marker=marker,
        ax=ax,
    )
    if log:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    if save:
        fig.savefig(_unique_filename(title), dpi=300, bbox_inches="tight")
    plt.show()


def plot_separate_by_group(
    group_by,
    df,
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    plot_by=None,
    marker=None,
    log=False,
    font_size=14,
    col_wrap=3,
    sharey=True,
    xlim=None,
    ylim=None,
    save=False,
    palette=None,
):
    """Plot one subplot per group (faceted layout).

    Args:
        group_by: Column name to facet by.
        df: DataFrame containing the data.
        x_col: Column for the x-axis.
        y_col: Column for the y-axis.
        x_label: x-axis label.
        y_label: y-axis label.
        title: Overall figure title.
        plot_by: Column for individual lines within a subplot (disables aggregation).
        marker: Matplotlib marker style.
        log: If True, use a log scale on the y-axis.
        font_size: Base font size.
        col_wrap: Number of subplot columns.
        sharey: If True, all subplots share the same y-axis scale.
        xlim: (min, max) tuple for x-axis limits.
        ylim: (min, max) tuple for y-axis limits.
        save: If True, save the figure as a PNG.
        palette: List of colors; cycles by group index.
    """
    required = {x_col, y_col, group_by}
    if plot_by is not None:
        required.add(plot_by)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Plot data is missing columns: {sorted(missing)}")

    plt.rcParams.update({"font.size": font_size})
    levels = list(pd.unique(df[group_by]))
    n = len(levels)
    if n == 0:
        raise ValueError(f"No groups found for '{group_by}'")

    cols = max(1, int(col_wrap))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=sharey)
    try:
        axes_list = list(axes.flat)
    except AttributeError:
        axes_list = [axes]

    for i, lvl in enumerate(levels):
        ax_i = axes_list[i]
        sub = df[df[group_by] == lvl]
        sns.lineplot(
            data=sub,
            x=x_col,
            y=y_col,
            units=plot_by if plot_by is not None else None,
            estimator=None if plot_by is not None else "mean",
            lw=1.2,
            alpha=0.8,
            marker=marker,
            color=palette[i % len(palette)] if palette is not None else None,
            ax=ax_i,
        )
        if log:
            ax_i.set_yscale("log")
        if xlim is not None:
            ax_i.set_xlim(xlim)
        if ylim is not None:
            ax_i.set_ylim(ylim)
        ax_i.set_xlabel(x_label)
        ax_i.set_ylabel(y_label)
        ax_i.set_title(str(lvl))

    for j in range(n, len(axes_list)):
        axes_list[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(title, y=0.995)
    if save:
        fig.savefig(_unique_filename(title), dpi=300, bbox_inches="tight")
    plt.show()


def _unique_filename(title: str) -> str:
    """Return a unique .png filename derived from title, avoiding overwrites."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", title.strip()) + ".png"
    name, ext = os.path.splitext(base)
    path, i = base, 1
    while os.path.exists(path):
        path = f"{name}_{i}{ext}"
        i += 1
    return path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_grouped(group_by, df, x_col, y_col, x_label, y_label, title,
                 plot_by=None, 
                 marker=None, log=False, font_size=14, xlim=None, ylim=None):
  
    # validate required columns
    required = {x_col, y_col, group_by}
    if plot_by is not None:
        required.add(plot_by)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Plot data is missing columns: {sorted(missing)}")

    plt.rcParams.update({'font.size': font_size})

    # prepare axes
    created_fig = False
    fig, ax = plt.subplots(figsize=(10, 6))

        #  individual series in one axes, colored by group_by
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=group_by,
        units=plot_by if plot_by is not None else None,
        estimator=None if plot_by is not None else "mean",
        lw=1.2,
            alpha=0.7,
            marker=marker,
            ax=ax,
        )

    if log:
        ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()



def plot_separate_by_group(group_by, df, x_col, y_col, x_label, y_label, title,
                           plot_by=None, marker=None, log=False, font_size=14,
                           col_wrap=3, sharey=True, palette=None, xlim=None, ylim=None):
    # validate required columns
    required = {x_col, y_col, group_by}
    if plot_by is not None:
        required.add(plot_by)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Plot data is missing columns: {sorted(missing)}")

    plt.rcParams.update({'font.size': font_size})

    levels = list(pd.unique(df[group_by]))
    n = len(levels)
    if n == 0:
        raise ValueError(f"No groups found for '{group_by}'")

    import math
    cols = max(1, int(col_wrap))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=sharey)
    if isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        try:
            axes_list = list(axes.flat)
        except Exception:
            axes_list = [axes]

    if palette is None:
        base_palette = sns.color_palette(n_colors=n)
        palette_map = {lvl: base_palette[i % len(base_palette)] for i, lvl in enumerate(levels)}
    elif isinstance(palette, dict):
        palette_map = palette
    else:
        palette_map = {lvl: palette[i % len(palette)] for i, lvl in enumerate(levels)}

    # Plot each group's lines separately
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
            color=palette_map.get(lvl),
            ax=ax_i,
        )

        if log:
            ax_i.set_yscale('log')
        if xlim is not None:
            ax_i.set_xlim(xlim)
        if ylim is not None:
            ax_i.set_ylim(ylim)

        ax_i.set_xlabel(x_label)
        ax_i.set_ylabel(y_label)
        ax_i.set_title(f"{lvl}")

    # Clear any unused axes
    for j in range(n, len(axes_list)):
        axes_list[j].set_visible(False)

    # One overall title
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(title, y=0.995)
    plt.show()
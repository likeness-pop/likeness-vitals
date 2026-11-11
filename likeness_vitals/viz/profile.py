import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

def profile_annotated_tilemap(
    df,
    cmap_name,
    alpha_min=0.15,
    alpha_max=1.0,
    figsize=(10, 10),
    plot_title=None,
    legend_title=None,
):
    """
    Plot an annotated tilemap of population profiles.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, with observations
        (e.g., population segments, neighborhoods)
        on the rows, and features on the columns.
    cmap_name : str
        Name of Matplotlib colormap to use.
    alpha_min : float
        Alpha value used to represent minimum `df` value.
    alpha_max : float
        Alpha value used to represent maximum `df` value.
    figsize : tuple
        Width and height of figure in inches.
    plot_title : str, None
        Plot title text.
    legend_title: str, None
        Legend title text.
    """
    # palette: one color per row
    pal = list(plt.colormaps[cmap_name].colors)[: len(df.index)][::-1]

    vals = df.to_numpy()
    nrows, ncols = vals.shape

    # normalize values -> alpha
    vmin = vals.min()
    vmax = vals.max()

    if vmax > vmin:
        alpha_vals = alpha_min + (vals - vmin) / (vmax - vmin) * (alpha_max - alpha_min)
    else:
        alpha_vals = np.full_like(vals, alpha_max, dtype=float)

    # build RGBA image: each row gets one base color, alpha varies by value
    rgba = np.zeros((nrows, ncols, 4))
    for i in range(nrows):
        r, g, b = mpl.colors.to_rgb(pal[i])
        rgba[i, :, 0] = r
        rgba[i, :, 1] = g
        rgba[i, :, 2] = b
        rgba[i, :, 3] = alpha_vals[i, :]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, aspect="auto", interpolation="none")

    # ticks / labels
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(df.index)

    # cell borders
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # maximum by column
    col_max = df.max(axis=0)

    # annotate cells
    for i in range(nrows):
        for j in range(ncols):
            val = df.iloc[i, j]
            is_max = val == col_max.iloc[j]

            # approximate displayed cell color 
            # after alpha blending onto white background
            base_rgb = np.array(mpl.colors.to_rgb(pal[i]))
            a = alpha_vals[i, j]
            blended_rgb = a * base_rgb + (1 - a) * np.array([1, 1, 1])

            luminance = (
                0.2126 * blended_rgb[0]
                + 0.7152 * blended_rgb[1]
                + 0.0722 * blended_rgb[2]
            )
            text_color = "white" if luminance < 0.5 else "black"

            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold" if is_max else "normal",
            )

    if plot_title is not None:
        ax.set_title(plot_title)

    # optional alpha legend
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "alpha_legend", [(1, 1, 1), (0.2, 0.2, 0.2)]
        ),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    if legend_title is not None:
        cbar.set_label(legend_title)

    plt.tight_layout()
    plt.show()

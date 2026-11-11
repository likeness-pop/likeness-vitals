import geopandas as gpd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Circle
import numpy as np
import pandas as pd


def hub_and_spoke_glyph_plot(
    gdf,
    value_cols,
    colors=None,
    symbol_size=0.5,
    alpha=0.9,
    figsize=(10, 10),
    use_representative_point=True,
    start_angle=0,
    clockwise=False,
    hub_radius=0.18,
    spoke_min=0.0,
    spoke_scale=0.82,
    spoke_linewidth=2.0,
    draw_hub=True,
    hub_facecolor="white",
    hub_edgecolor="black",
    hub_linewidth=0.8,
    normalize=True,
    show_tip_markers=False,
    marker_size=12,
    legend_loc="upper left",
    legend_title="Categories",
    choropleth_col=None,
    choropleth_alpha=0.5,
):
    """
    Plot polygons with line-spoke glyphs at representative points / centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Base map boundaries.
    value_cols : list[str]
        Columns used for spoke lengths.
    colors : list-like, optional
        One color per value column.
    symbol_size : float
        Glyph size in inches.
    alpha : float
        Spoke transparency.
    figsize : tuple
    use_representative_point : bool
        If True use representative_point(), else centroid.
    start_angle : float
        Rotation in degrees.
    clockwise : bool
        If True, spokes proceed clockwise.
    hub_radius : float
        Radius of center hub in normalized glyph coordinates.
    spoke_min : float
        Minimum spoke start radius. Usually 0 if spokes start at hub edge after drawing.
    spoke_scale : float
        Maximum added spoke length beyond the hub region.
    spoke_linewidth : float
        Width of spoke lines.
    normalize : bool
        If True, normalize each row to proportions summing to 1.
        If False, scale by global max.
    show_tip_markers : bool
        If True, draw dots at spoke ends.
    marker_size : float
        Tip marker size.
    legend_loc : str
        Location of legend on plot.
    legend_title : str
        Legend title.
    choropleth_col : str, None
        Name of column in `gdf` used to (optionally) shade the base map boundaries.
    choropleth_alpha : float
        Opacity level for choropleth shading.
    """

    gdf = gdf.copy()

    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS.")
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    pts = (
        gdf.geometry.representative_point()
        if use_representative_point
        else gdf.geometry.centroid
    )
    gdf["x"] = pts.x
    gdf["y"] = pts.y

    if colors is None:
        colors = plt.cm.tab10.colors[: len(value_cols)]
    if len(colors) < len(value_cols):
        raise ValueError("Need at least as many colors as value columns.")

    vals = gdf[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    if not normalize:
        global_max = vals.to_numpy().max()
        if global_max <= 0:
            global_max = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    if choropleth_col is None:
        gdf.plot(ax=ax, facecolor="whitesmoke", edgecolor="black", linewidth=0.8)
    else:
        gdf.plot(
            ax=ax,
            column=choropleth_col,
            cmap="Grays",
            edgecolor="black",
            alpha=choropleth_alpha,
            linewidth=0.8,
            legend=True,
            legend_kwds={
                "orientation": "horizontal",
                "shrink": 0.4,
                "pad": 0.01,
                "alpha": choropleth_alpha,
                "label": choropleth_col,
            },
        )

    n = len(value_cols)
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.deg2rad(start_angle)

    if clockwise:
        base_angles = -base_angles

    fig_w, fig_h = fig.get_size_inches()
    w_frac = symbol_size / fig_w
    h_frac = symbol_size / fig_h

    for _, row in gdf.iterrows():
        values = (
            pd.to_numeric(row[value_cols], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

        if np.all(values <= 0):
            continue

        if normalize:
            total = values.sum()
            if total <= 0:
                continue
            scaled = values / total
        else:
            scaled = values / global_max

        disp = ax.transData.transform((row["x"], row["y"]))
        fig_xy = fig.transFigure.inverted().transform(disp)

        # Use a plain Cartesian inset, easier for custom spoke drawing
        iax = fig.add_axes(
            [fig_xy[0] - w_frac / 2, fig_xy[1] - h_frac / 2, w_frac, h_frac]
        )
        iax.set_xlim(-1, 1)
        iax.set_ylim(-1, 1)
        iax.set_aspect("equal")
        iax.axis("off")
        iax.set_facecolor("none")

        # Draw spokes
        for ang, val, col in zip(base_angles, scaled, colors[:n]):
            r0 = max(spoke_min, hub_radius if draw_hub else 0.0)
            r1 = r0 + spoke_scale * val

            x0, y0 = r0 * np.cos(ang), r0 * np.sin(ang)
            x1, y1 = r1 * np.cos(ang), r1 * np.sin(ang)

            iax.plot(
                [x0, x1],
                [y0, y1],
                color=col,
                linewidth=spoke_linewidth,
                alpha=alpha,
                solid_capstyle="round",
                zorder=2,
            )

            if show_tip_markers:
                iax.scatter([x1], [y1], s=marker_size, color=col, alpha=alpha, zorder=3)

        if draw_hub:
            hub = Circle(
                (0, 0),
                radius=hub_radius,
                facecolor=hub_facecolor,
                edgecolor=hub_edgecolor,
                linewidth=hub_linewidth,
                zorder=4,
            )
            iax.add_patch(hub)

    handles = [
        Patch(facecolor=c, edgecolor="none", alpha=alpha, label=col)
        for c, col in zip(colors[: len(value_cols)], value_cols)
    ]
    ax.legend(handles=handles, title=legend_title, loc=legend_loc)
    ax.set_axis_off()
    return fig, ax


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

            # approximate displayed cell color after alpha blending onto white background
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

import geopandas as gpd
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
    gdf : GeoDataFrame
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

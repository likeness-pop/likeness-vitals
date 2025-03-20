"""Spatial & Geometric Operations
"""

import contextlib

import geopandas
import numpy
import pandas
from scipy.spatial import cKDTree
from shapely import Point, Polygon

__author__ = "jGaboardi"


__all__ = [
    "disaggregate",
    "generate_points",
    "synthetic_locations",
]


def disaggregate(
    df_: pandas.DataFrame, cnt_col: str, id_col: str = None
) -> pandas.DataFrame:
    """Disaggregate tabular weighted records. When no
    ID column specified, the IDs will not be extended.

    Parameters
    ----------
    df_ : pandas.DataFrame
        Aggregated person records.
    cnt_col : str
        Counts column name.
    id_col : str (default None)
        ID column name.

    Parameters
    ----------
    df_ : pandas.DataFrame
        Disggregated person records.
    """

    cols = df_.columns
    df_ = pandas.DataFrame(numpy.repeat(df_.values, df_[cnt_col], axis=0))
    df_.columns = cols
    df_[cnt_col] = 1
    if id_col:
        df_[id_col] = df_[id_col] + "-" + df_.index.astype(str)
    return df_


def _param_checker(minsep: int, maxsep: int, maxiter: int) -> bool:
    """ "Check point generation parameters."""

    if minsep < 0:
        raise ValueError(f"``minsep`` can't be negative: {minsep}.")
    if maxsep < 0:
        raise ValueError(f"``maxsep`` can't be negative: {maxsep}.")
    if maxsep < minsep:
        raise ValueError(
            f"``maxsep < minsep``: ({maxsep} < {minsep}). "
            "Maximum seperation must be >= minimum seperation."
        )
    if maxiter < 1:
        raise ValueError(f"``maxiter`` must be 1 or greater: {maxiter}.")

    return True


def generate_points(
    npoints: int,
    polygon: Polygon,
    seed: int,
    minsep: float | float,
    maxsep: float | float,
    maxiter: int,
    params_checked: bool = False,
) -> list:
    """Generate points within a polygon.

    Parameters
    ----------
    npoints : int
        Point count to generate.
    polygon : Polygon
        Polygon in which to generate points.
    seed : int
        Random state for ``numpy.random``.
    minsep : int | float
        Minimum separation distance between points.
    maxsep : int | float
        Maximum separation distance between points.
    maxiter : int
        Iterations to run before relaxing ``minsep`` and ``maxsep``.
    params_checked : bool = False
        Have point generation parameters already been verified?

    Returns
    -------
    points : list
        Points within a polygon.
    """

    # ensure point generations arguments validity
    if not params_checked:
        _param_checker(minsep, maxsep, maxiter)

    points = []
    minx, miny, maxx, maxy = polygon.bounds
    itercount, _maxiter = 0, maxiter
    rng = numpy.random.default_rng(seed).uniform
    while len(points) < npoints:
        itercount += 1
        point = Point(rng(low=minx, high=maxx), rng(low=miny, high=maxy))
        if polygon.contains(point):
            # enforce a min seperation dist unless proving too difficult
            if maxiter > itercount:
                mns, mxs = minsep, maxsep
            else:
                # grow the acceptable min/max sep if needed
                maxiter += _maxiter
                minsep, maxsep = minsep / 1.5, maxsep * 1.5
                mns, mxs = minsep, maxsep
            if points and mns and mxs:
                all_coords = numpy.array([(p.x, p.y) for p in points])
                curr_coords = numpy.array([(point.x, point.y)])
                ckdtree = cKDTree(curr_coords).query(all_coords, k=1)
                curr_min, curr_max = ckdtree[0].min(), ckdtree[0].max()
                if curr_min < mns or curr_max > mxs:
                    continue
            points.append(point)
    return points


def synthetic_locations(
    pnt_df: pandas.DataFrame,
    pgn_gdf: geopandas.GeoDataFrame,
    geom_id: str,
    seed: int = 0,
    minsep: int | float = 10,
    maxsep: int | float = 20,
    maxiter: int = 100,
) -> geopandas.GeoDataFrame:
    """Generate a set number of synthetic locations within polygons.

    Parameters
    ----------
    pnt_gdf : pandas.DataFrame
        Tabular records for generating points.
    pgn_gdf : geopandas.GeoDataFrame
        Polygons to generate points within.
    geom_id : str
        Polygon ID for groupby.
    seed: int (default 0)
        Random state for ``numpy.random``.
    minsep : int | float (default 10)
        Minimum separation distance between points.
    maxsep : int | float (default 20)
        Maximum separation distance between points.
    maxiter : int (default 100)
        Iterations to run before relaxing ``minsep`` and ``maxsep``.

    Returns
    -------
    geopandas.GeoDataFrame
        Generated points for tabular records.
    """

    # set point generations arguments and ensure validity
    pnt_kws = {"params_checked": _param_checker(minsep, maxsep, maxiter)}

    with contextlib.suppress(KeyError):
        pgn_gdf = pgn_gdf.set_index(geom_id)

    pnts = []
    _df = pnt_df.sort_values(geom_id)
    for ix, _dfx in _df.groupby(geom_id):
        seed += 1
        polygon = pgn_gdf.geometry.loc[ix]
        npnt = _dfx.shape[0]
        _pnts = generate_points(npnt, polygon, seed, minsep, maxsep, maxiter, **pnt_kws)
        pnts.extend(_pnts)

    return geopandas.GeoDataFrame(_df, geometry=pnts, crs=pgn_gdf.crs)

"""Shared utility functionality for Likeness modules
"""

import pathlib
import time
import uuid
import warnings
from collections.abc import Iterable
from functools import wraps
from typing import Any

import geopandas
import pandas
import tqdm
from tqdm.auto import tqdm as tqdm_auto


def function_timer(wrapped_function: callable) -> callable:
    """This can be used as a wrapper. For example:

        ```
        @function_timer
        def some_func(x):
            return x**2
        ```

    This will print the elapsed time in minutes.

    """

    @wraps(wrapped_function)
    def wrapper(*args, **kwargs) -> Any:
        fname = wrapped_function.__name__
        t1 = time.time()
        _wrapper = wrapped_function(*args, **kwargs)
        t2 = time.time()
        total = round((t2 - t1) / 60.0, 5)
        print(f"\t{total} min. -- ``{fname}()``")
        return _wrapper

    return wrapper


def progress(iterable_object: Iterable, desc: str) -> tqdm.asyncio.tqdm_asyncio:
    """Progress bar for iterators.

    Parameters
    ----------
    iterable_object : Iterable
        Any iterable object with which to apply a progress bar.
    desc : str
        User provided description to add to progress bar.

    Returns
    -------
    tqdm.asyncio.tqdm_asyncio
        Progress bar object over which to be iterated.
    """

    return tqdm_auto(iterable_object, desc=desc)


def match(
    x1: pandas.DataFrame | geopandas.GeoDataFrame,
    x2: pandas.Series | pandas.DataFrame | geopandas.GeoSeries | geopandas.GeoDataFrame,
    on: None | str = None,
    v: None | str = None,
    strict: bool = False,
) -> pandas.Series:
    """Matches values between DataFrames based on a common key.

    Parameters
    ----------
    x1 : pandas.DataFrame | geopandas.GeoDataFrame
        Target data.
    x2 : pandas.Series | pandas.DataFrame | geopandas.GeoSeries | geopandas.GeoDataFrame
        Source data.
    on : str (default None)
        Common key between ``df1`` and ``df2``. If ``None``, the
        ``df1`` index is used.
    v : str (default None)
        Variable in ``df2`` whose values will be matched to ``df1``.
        If ``x2`` is a DataFrame, but v is not provided, defaults
        to the first variable after ``on``.
    strict : bool (default False)
        Ensure zipped iterables are the same length.

    Returns
    -------
    out : pandas.Series
        Values of ``v`` in ``df2`` matched to ``df1``.
    """

    # type checking
    pd_frame = isinstance(x2, pandas.DataFrame)
    gpd_frame = isinstance(x2, geopandas.GeoDataFrame)

    pd_series = isinstance(x2, pandas.Series)
    gpd_series = isinstance(x2, geopandas.GeoSeries)

    # create match contingency
    if pd_frame or gpd_frame:
        assert x2.shape[1] >= 2, "Source data must have at least two columns."
        if v is None:
            v = x2.columns[x2.columns != on][0]
        val_match = dict(zip(x2[on], x2[v], strict=strict))

    elif pd_series or gpd_series:
        val_match = dict(zip(x2.index.values, x2.values, strict=strict))

    else:
        raise TypeError(f"{type(x2)} not supported for ``x2``.")

    # map values
    out = x1[on].map(val_match) if on is not None else x1.index.map(val_match)

    return out


def get_censusapikey(path: str | pathlib.Path = "") -> str:
    """Fetch your Census API key. See README.md for more details.

    Parameters
    ----------
    path : str | pathlib.Path (default '')
        Path to directory where ``'censusapikey.txt'`` is stored.
        The path can be absolute (full/path/to/file/) or relative
        (../).

    Returns
    -------
    key : str
        Census API key.
    """

    key_file = "censusapikey.txt"
    if isinstance(path, str):
        path = pathlib.Path(path)
    file_path = path / key_file
    if file_path.exists():
        with open(file_path) as f:
            key = f.readlines()[0].replace("\n", "")

    else:
        key = None
        msg = (
            f"No key file ('{key_file}') found in the following directory: "
            f"'{path}'. Check that you entered the correct "
            f"``path`` and try again. Returning ``{key}``."
        )
        warnings.warn(msg, stacklevel=2)

    return key


def create_uid(
    df: pandas.DataFrame | geopandas.GeoDataFrame,
    id_name: str,
    use_index: bool = False,
    from_columns: None | list = None,
    set_index: bool = False,
    drop_cols: bool = False,
    breaker: str = "_",
) -> pandas.DataFrame | geopandas.GeoDataFrame:
    """Generate a unique identifying ID.

    Parameters
    ----------
    df : pandas.DataFrame | geopandas.GeoDataFrame
        Input data.
    id_name : str
        Name of the new ID.
        * set to ``'uuid'`` to generate a Universally Unique Identifier
        with the ``uuid.uuid1().hex`` algorithm. See *Notes* below.
    use_index : bool (default False)
        Include the original index values in the new unique ID.
    from_columns : list | None (default (None)
        Use a concatenation of these columns to create the ID.
    set_index : bool (default False)
        Set the newly generated ID as the index.
    drop_cols : bool (default False)
        Drop intermediary columns if ``from_columns`` is specified.
    breaker : str (default '_')
        Break up components of the unique ID if ``from_columns`` is specified.

    Returns
    -------
    df : pandas.DataFrame | geopandas.GeoDataFrame
        Input data with new ID.

    Notes
    -----
    See https://docs.python.org/3/library/uuid.html#uuid.uuid1
    """

    def _idx_generator(ix: int) -> str:
        """concatenate specified column values."""
        return breaker.join([str(df.loc[ix, c]) for c in from_columns])

    if not set_index and drop_cols:
        raise RuntimeError(
            f"``set_index``=={set_index} and"
            f"``drop_cols``=={drop_cols}. Must change configuation."
        )

    # either generate a true UUID
    generate_uuid = False
    if id_name.lower() == "uuid":
        generate_uuid = True
        unique_id = [uuid.uuid1().hex for _ in df.index]

    # or create an ID based on other variable values
    if not generate_uuid:
        if isinstance(from_columns, str):
            from_columns = [from_columns]

        if use_index:
            orig_index = df.index.copy()
            index_name = "index" if not df.index.name else df.index.name
            df.reset_index(inplace=True)
            from_columns += [index_name]

        if len(from_columns) > 1:
            unique_id = df.index.map(_idx_generator)
        else:
            unique_id = df[from_columns]

        if use_index:
            df.index = orig_index
            df.drop(columns=index_name, inplace=True)
            from_columns.pop()

        if drop_cols:
            df.drop(columns=from_columns, inplace=True)

    df[id_name] = unique_id

    if set_index:
        df.set_index(id_name, inplace=True)

    return df

"""Shared utility functionality for Likeness modules"""

# ruff: noqa: E402

import contextlib
import warnings
from importlib.metadata import PackageNotFoundError, version

# warnings.warn(
#     (
#         "The ``likeness-vitals`` package is transitioning to a modular API for a "
#         "cleaner top-level namespace. Classes, functions, and attributes that "
#         "are currently available in the top-level will only be available within "
#         "their home modules. Both the current and the updated API will be in place "
#         "for at least two minor releases following the inclusion of this warning. "
#         "The release at the time of this warning inclusion was ``v1.3.0``, so "
#         "the current API will be stable until ``v1.5.0`` at the soonest, and "
#         "the ``v2.0.0`` release will be the absolute deadline. "
#     ),
#     FutureWarning,
#     stacklevel=1,
# )
#
from . import constants, sg_ops, vitals
from .constants import (
    BGID,
    BKID,
    CNT,
    EPSG_3857,
    EPSG_4326,
    GID,
    HID,
    PID,
    SCL,
    TRS,
    XID,
)
from .sg_ops import disaggregate, generate_points, synthetic_locations
from .vitals import (
    create_uid,
    function_timer,
    get_censusapikey,
    match,
    progress,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("likeness_vitals")

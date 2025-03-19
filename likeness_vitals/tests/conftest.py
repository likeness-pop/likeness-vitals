import ast
import time
from sys import platform

import pytest

import likeness_vitals

# --------------------------------------------------------------
# adding command line options
# ---------------------------


def pytest_addoption(parser):
    """Add custom command line arguments to the testing suite"""

    # flag for local or remote/VM testing
    parser.addoption(
        "--local",
        action="store",
        default="True",
        help="Boolean flag for local or remote/VM testing.",
        choices=("True", "False"),
        type=str,
    )

    # flag for `dev` environment testing (bleeding edge dependencies)
    parser.addoption(
        "--env",
        action="store",
        default="latest",
        help=(
            "Environment type label of dependencies for determining whether certain "
            "tests should be run. Generally we are working with minimum/oldest, "
            "latest/stable, and bleeding edge."
        ),
        type=str,
    )

    # flag for determining package manager used in testing
    parser.addoption(
        "--package_manager",
        action="store",
        default="micromamba",
        help=(
            "Package manager label of dependencies for determining "
            "whether certain tests should be run."
        ),
        type=str,
    )


# --------------------------------------------------------------
# declaring attributes & methods for the configuration
# ----------------------------------------------------


@likeness_vitals.vitals.function_timer
def timing_function(wait_time: int) -> int:
    """Simple helper for wait & wrapper testing."""
    time.sleep(wait_time)
    return wait_time


# --------------------------------------------------------------
# adding accessible attributes & methods to the configuration
# -----------------------------------------------------------


def pytest_configure(config):
    """Set session attributes."""

    # ------------------------------------------------------
    # declaring command line options as attributes
    # --------------------------------------------

    # ``local`` from ``pytest_addoption()``
    pytest.LOCAL = ast.literal_eval(config.getoption("local"))

    # ``env`` from ``pytest_addoption()``
    pytest.ENV = config.getoption("env")
    valid_env_suffix = ["min", "latest", "dev"]
    assert pytest.ENV.split("_")[-1] in valid_env_suffix

    # ``package_manager`` from ``pytest_addoption()``
    pytest.PACKAGE_MANAGER = config.getoption("package_manager")
    valid_package_managers = ["conda", "micromamba"]
    assert pytest.PACKAGE_MANAGER in valid_package_managers

    # ------------------------------------------------------
    # declaring custom attributes and methods
    # ---------------------------------------

    # grouped tests with ``pytest.xdist``
    pytest.xdist_group_1 = pytest.mark.xdist_group(name="xdist_group_1")
    pytest.xdist_group_2 = pytest.mark.xdist_group(name="xdist_group_2")

    # on windows?
    pytest.SYS_WIN = platform.startswith("win")

    # add a help function as an attribute
    pytest.timing_function = timing_function


# --------------------------------------------------------------
# run stuff before testing suite starts
# -------------------------------------


def pytest_sessionstart(session):  # noqa: ARG001
    """Do this before starting tests."""
    time.sleep(1)


# --------------------------------------------------------------
# run stuff after testing suite finishes
# --------------------------------------


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Do this after all tests are finished."""

    # specifically for use with ``pytest-xdist``
    if not hasattr(session.config, "workerinput"):
        pass

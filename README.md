# Likeness-Vitals: Shared utility functionality for the Likeness ecosystem

![tag](https://img.shields.io/github/v/release/likeness-pop/likeness-vitals?include_prereleases&sort=semver)
[![PyPI version](https://badge.fury.io/py/likeness-vitals.svg)](https://badge.fury.io/py/likeness-vitals)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/likeness_vitals.svg)](https://anaconda.org/conda-forge/likeness_vitals)

[![Continuous Integration](https://github.com/likeness-pop/likeness-vitals/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/likeness-pop/likeness-vitals/actions/workflows/continuous_integration.yml)
[![codecov](https://codecov.io/gh/likeness-pop/likeness-vitals/branch/develop/graph/badge.svg)](https://codecov.io/gh/likeness-pop/likeness-vitals)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Installation

### Conda-forge (recommended)

The `likeness-vitals` feedstock is available via the [conda-forge channel](https://github.com/conda-forge/likeness_vitals-feedstock).

```
$ conda install --channel conda-forge likeness_vitals
```

### PyPI

`likeness-vitals` is available on the [Python Package Index](https://pypi.org/project/likeness-vitals/).

```
$ pip install likeness_vitals
```

### Source

#### Directly via GitHub + `pip`

```
$ pip install git+https://github.com/likeness-pop/likeness-vitals.git@develop
```

#### Download + `pip`

Download the source distribution (``.tar.gz``) and decompress where desired. From that location:

```
$ pip install .
```

## Development

1. Clone the repository to the desired location.
2. Install in editable mode
   * Navigate to where the repo was cloned locally.
   * Within that directory:
      ```
      $ pip install -e .
      ```
3. Open an Issue for discussion
4. In a branch off `develop`, implement update/bug fix/etc.
5. Create a minimal Pull Request with clear description linked back to the associated issue from (3.)
6. Wait for review from maintainers
7. Adjust as directed
8. Once merged, fetch down `origin/develop` and merge into the local `develop`
9. Delete the branch created in (4.)
10. Start over at (2.)

## Ecosystem-level conda environments

The conda environments provided in `./envs/*` contain all dependencies required to use `livelike`, `pymedm` / `pmedm-legacy`, and `likeness-vitals`.

The install script will create a Python 3.14 environment and automatically choose between the vanilla and CUDA environment based on system type. To run it:

```
bash setup.sh
```

If an older version of Python is desired, simply update `setup.sh` appropriately.

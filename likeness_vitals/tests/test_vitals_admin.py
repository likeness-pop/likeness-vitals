import pandas
import pytest

import likeness_vitals


class TestMatch:
    def setup_method(self):
        # target data with key as column
        self.a1 = pandas.DataFrame({"id": ["A", "A", "A", "B", "C", "C"]})
        # source data as dataframe
        self.b1 = pandas.DataFrame({"id": ["A", "B", "C"], "val": [1, 2, 3]})
        # source data as series
        self.b2 = pandas.Series([1, 2, 3], name="val", index=["A", "B", "C"])

    def test_match_df_df(self):
        known = [1, 1, 1, 2, 3, 3]
        observed = likeness_vitals.vitals.match(self.a1, self.b1, on="id").tolist()
        assert observed == known

    def test_match_df_series(self):
        known = [1, 1, 1, 2, 3, 3]
        observed = likeness_vitals.vitals.match(self.a1, self.b2, on="id").tolist()
        assert observed == known

    def test_match_df_type_error(self):
        x2 = "two"
        with pytest.raises(TypeError, match=f"{type(x2)} not supported for ``x2``."):
            likeness_vitals.vitals.match("one", x2)


@pytest.xdist_group_1
def test_vitals_function_timer():
    known = 3
    observed = pytest.timing_function(known)
    assert known == observed


@pytest.xdist_group_2
def test_vitals_progress():
    known = 6
    observed = 0
    for _i in likeness_vitals.vitals.progress(range(2), "progress test"):
        observed += pytest.timing_function(3)
    assert known == observed


@pytest.xdist_group_1
def test_census_api_key_not_found():
    """We can only really check the 'not found' situation here."""
    known = None
    with pytest.warns(UserWarning, match="No key file."):
        observed = likeness_vitals.vitals.get_censusapikey()
    assert known == observed


#################################################################################
#################################################################################

# The following is a example test demonstrating working with
# ``pytest_addoption`` argument variables as a ``fixture`` (``local``) and
# as an attribute (``ENV``) declared in ``conftest.py``


def test_param_options_example():
    local = pytest.LOCAL
    is_dev = pytest.ENV.endswith("dev")

    minuend = 20 if local else 10
    subtrahend = 10 if is_dev else 5
    difference = minuend - subtrahend

    if local and not is_dev:
        # local testing -- non-bleeding edge environment
        assert difference == 15

    elif local and is_dev:
        # local testing -- bleeding edge environment
        assert difference == 10

    elif not local and not is_dev:
        # non-local testing -- non-bleeding edge environment
        assert difference == 5

    else:
        # non-local testing -- bleeding edge environment
        assert difference == 0


#################################################################################

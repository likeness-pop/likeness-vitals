import numpy
import pandas
import pytest

import likeness_vitals


@pytest.fixture
def df() -> pandas.DataFrame:
    """synthetic input records"""
    return pandas.DataFrame(
        {"c1": ["A", "B", "C"], "c2": ["x", "x", "y"], "vals": [10, 20, 30]},
        index=[999, 998, 997],
    )


class TestVitalsUUID:
    def test_no_index(self, df):
        """UUID as a column"""
        _df = likeness_vitals.create_uid(df, "uuid")
        known = numpy.array(
            [["A", "x", 10], ["B", "x", 20], ["C", "y", 30]], dtype=object
        )
        observed = _df[[c for c in df.columns if c != "uuid"]].copy()
        numpy.testing.assert_array_equal(known, observed)

        known = 3
        observed = len(set(_df["uuid"]))
        assert known == observed

        known = numpy.array([999, 998, 997])
        observed = _df.index.values
        numpy.testing.assert_array_equal(known, observed)

    def test_with_index(self, df):
        """UUID as the index"""
        _df = likeness_vitals.create_uid(df, "uuid", set_index=True)
        known = numpy.array(
            [["A", "x", 10], ["B", "x", 20], ["C", "y", 30]], dtype=object
        )
        observed = _df.values
        numpy.testing.assert_array_equal(known, observed)

        known = 3
        observed = len(set(df.index))
        assert known == observed


class TestVitalsNonUUID:
    def test_1col(self, df):
        """single column; no index -- completely useless case"""
        _df = (
            likeness_vitals.create_uid(df, "id1", from_columns="c1")
            .reset_index()
            .values
        )
        known = numpy.array(
            [
                [999, "A", "x", 10, "A"],
                [998, "B", "x", 20, "B"],
                [997, "C", "y", 30, "C"],
            ],
            dtype=object,
        )
        observed = _df
        numpy.testing.assert_array_equal(known, observed)

    def test_2col(self, df):
        """multiple columns; no index"""
        _df = (
            likeness_vitals.create_uid(df, "id1", from_columns=["c1", "c2"])
            .reset_index()
            .values
        )
        known = numpy.array(
            [
                [999, "A", "x", 10, "A_x"],
                [998, "B", "x", 20, "B_x"],
                [997, "C", "y", 30, "C_y"],
            ],
            dtype=object,
        )
        observed = _df
        numpy.testing.assert_array_equal(known, observed)

    def test_1col_idx(self, df):
        """single column; use index"""
        _df = (
            likeness_vitals.create_uid(df, "id1", use_index=True, from_columns="c1")
            .reset_index()
            .values
        )
        known = numpy.array(
            [
                [999, "A", "x", 10, "A_999"],
                [998, "B", "x", 20, "B_998"],
                [997, "C", "y", 30, "C_997"],
            ],
            dtype=object,
        )
        observed = _df
        numpy.testing.assert_array_equal(known, observed)

    def test_2col_idx(self, df):
        """multiple columns; use index"""
        _df = (
            likeness_vitals.create_uid(
                df, "id1", use_index=True, from_columns=["c1", "c2"]
            )
            .reset_index()
            .values
        )
        known = numpy.array(
            [
                [999, "A", "x", 10, "A_x_999"],
                [998, "B", "x", 20, "B_x_998"],
                [997, "C", "y", 30, "C_y_997"],
            ],
            dtype=object,
        )
        observed = _df
        numpy.testing.assert_array_equal(known, observed)

    def test_2col_drop(self, df):
        """multiple columns; set index; drop columns"""
        _df = (
            likeness_vitals.create_uid(
                df,
                "id1",
                set_index=True,
                drop_cols=True,
                from_columns=["c1", "c2"],
            )
            .reset_index()
            .values
        )
        known = numpy.array(
            [["A_x", 10], ["B_x", 20], ["C_y", 30]],
            dtype=object,
        )
        observed = _df
        numpy.testing.assert_array_equal(known, observed)

    def test_2col_idx_drop(self, df):
        """multiple columns; use index; set index; drop columns"""
        _df = (
            likeness_vitals.create_uid(
                df,
                "id1",
                set_index=True,
                drop_cols=True,
                use_index=True,
                from_columns=["c1", "c2"],
            )
            .reset_index()
            .values
        )
        known = numpy.array(
            [["A_x_999", 10], ["B_x_998", 20], ["C_y_997", 30]],
            dtype=object,
        )
        observed = _df
        numpy.testing.assert_array_equal(known, observed)

    def test_conflicting_kwargs_error(self, df):
        """conflicting keyword argument combination"""
        set_index = False
        drop_cols = True
        msg = (
            f"``set_index``=={set_index} and"
            f"``drop_cols``=={drop_cols}. Must change configuation."
        )
        with pytest.raises(RuntimeError, match=msg):
            likeness_vitals.create_uid(
                df, "id1", set_index=set_index, drop_cols=drop_cols
            )

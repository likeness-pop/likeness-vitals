import geopandas
import pandas
import pytest
import shapely

import likeness_vitals

gid = likeness_vitals.constants.GID
pid = likeness_vitals.constants.PID
cnt = likeness_vitals.constants.CNT


def _pnt_df() -> pandas.DataFrame:
    """point dataframe helper"""
    return pandas.DataFrame(
        {gid: ["A"] * 3 + ["B"] * 3, pid: [f"p{ix}" for ix in range(10, 16)]}
    )


@pytest.fixture
def pnt_df() -> pandas.DataFrame:
    """representation of agents"""
    return _pnt_df()


@pytest.fixture
def pnt_df_cnt() -> pandas.DataFrame:
    """representation of agents with weights"""
    return _pnt_df().assign(**{cnt: [1, 2, 3] * 2})


def _plg_df() -> geopandas.GeoDataFrame:
    """polygon dataframe helper"""
    pgons = [
        shapely.Polygon(((0, 10), (0, 10), (10, 10), (10, 0), (0, 0))),
        shapely.Polygon(((10, 0), (10, 10), (20, 10), (20, 0), (10, 0))),
    ]
    return geopandas.GeoDataFrame({gid: ["A", "B"], "geometry": pgons}).set_index(gid)


@pytest.fixture
def plg_df() -> geopandas.GeoDataFrame:
    """representation of areal units"""
    return _plg_df()


@pytest.fixture
def plg_cent_df() -> geopandas.GeoDataFrame:
    """centroid representation of areal units"""
    return _plg_df().pipe(lambda df: df.assign(**{"geometry": df.geometry.centroid}))


class TestVitalsDisAggLocs:
    @pytest.fixture(autouse=True)
    def setup_method(self, pnt_df_cnt):
        self.disagg_df = likeness_vitals.sg_ops.disaggregate(
            pnt_df_cnt, cnt, id_col=pid
        )

    def test_disagg_records(self):
        known = 12
        observed = self.disagg_df.shape[0]
        assert observed == known

    def test_disagg_count(self):
        known = 12
        observed = self.disagg_df[cnt].sum()
        assert observed == known

    def test_disagg_equal(self, pnt_df_cnt):
        known = pnt_df_cnt[cnt].sum()
        observed = self.disagg_df[cnt].sum()
        assert observed == known


class TestVitalsSynthLocs:
    @pytest.fixture(autouse=True)
    def setup_method(self, pnt_df_cnt, plg_df):
        self.disagg_df = likeness_vitals.sg_ops.disaggregate(
            pnt_df_cnt, cnt, id_col=pid
        )
        self.disagg_df = pandas.concat(
            [self.disagg_df[:2], self.disagg_df[6:8]], ignore_index=True
        )
        self.minsep = 0.2
        self.maxsep = 2
        self.synthlocs_df = likeness_vitals.sg_ops.synthetic_locations(
            self.disagg_df,
            plg_df,
            gid,
            minsep=self.minsep,
            maxsep=self.maxsep,
        )

    def test_respect_A(self):  # noqa: N802
        known = 0.4031967939355218
        a_locs = self.synthlocs_df[self.synthlocs_df[gid] == "A"].copy()
        observed = a_locs.loc[0].geometry.distance(a_locs.loc[1].geometry)

        respects = (observed >= self.minsep) and (observed <= self.maxsep)
        assert respects
        assert pytest.approx(observed) == known

    def test_respect_B(self):  # noqa: N802
        known = 1.7110617211172108
        b_locs = self.synthlocs_df[self.synthlocs_df[gid] == "B"].copy()
        observed = b_locs.loc[2].geometry.distance(b_locs.loc[3].geometry)

        respects = (observed >= self.minsep) and (observed <= self.maxsep)
        assert respects
        assert pytest.approx(observed) == known


class TestVitalsGeneratePnts:
    @pytest.fixture(autouse=True)
    def setup_method(self, plg_df):
        self.npoints = 2
        self.minsep = 0.2
        self.maxsep = 2
        self.genpnts = likeness_vitals.sg_ops.generate_points(
            self.npoints,
            plg_df.loc["A"].geometry,
            1,
            self.minsep,
            self.maxsep,
            100,
        )

    def test_npoints_generated(self):
        known = self.npoints
        observed = len(self.genpnts)
        assert observed == known

    def test_respect(self):
        known = 0.4031967939355218
        observed = self.genpnts[0].distance(self.genpnts[1])

        respects = (observed >= self.minsep) and (observed <= self.maxsep)
        assert respects
        assert pytest.approx(observed) == known


class TestVitalsPointGenErrors:
    def test_maxsep_lt_minsep_error(self):
        minsep = 10
        maxsep = 5
        with pytest.raises(ValueError, match="``maxsep < minsep``"):
            likeness_vitals.sg_ops.generate_points(None, None, None, minsep, maxsep, 10)

    def test_minsep_error(self):
        minsep = -1
        with pytest.raises(
            ValueError, match=f"``minsep`` can't be negative: {minsep}."
        ):
            likeness_vitals.sg_ops.generate_points(None, None, None, minsep, 10, 10)

    def test_maxsep_error(self):
        maxsep = -1
        with pytest.raises(
            ValueError, match=f"``maxsep`` can't be negative: {maxsep}."
        ):
            likeness_vitals.sg_ops.generate_points(None, None, None, 10, maxsep, 10)

    def test_maxiter_error(self):
        maxiter = 0
        with pytest.raises(
            ValueError, match=f"``maxiter`` must be 1 or greater: {maxiter}."
        ):
            likeness_vitals.sg_ops.generate_points(None, None, None, 10, 10, maxiter)


class TestVitalsSynthLocErrors:
    def test_maxsep_lt_minsep_error(self):
        minsep = 10
        maxsep = 5
        with pytest.raises(ValueError, match="``maxsep < minsep``"):
            likeness_vitals.sg_ops.synthetic_locations(
                None, None, None, minsep=minsep, maxsep=maxsep
            )

    def test_minsep_error(self):
        minsep = -1
        with pytest.raises(
            ValueError, match=f"``minsep`` can't be negative: {minsep}."
        ):
            likeness_vitals.sg_ops.synthetic_locations(None, None, None, minsep=minsep)

    def test_maxsep_error(self):
        maxsep = -1
        with pytest.raises(
            ValueError, match=f"``maxsep`` can't be negative: {maxsep}."
        ):
            likeness_vitals.sg_ops.synthetic_locations(None, None, None, maxsep=maxsep)

    def test_maxiter_error(self):
        maxiter = 0
        with pytest.raises(
            ValueError, match=f"``maxiter`` must be 1 or greater: {maxiter}."
        ):
            likeness_vitals.sg_ops.synthetic_locations(
                None, None, None, maxiter=maxiter
            )

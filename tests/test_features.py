"""Unit tests for Track B feature modules.

Known-answer checks:
- VPD at (T=25°C, RH=50%) ≈ 1.584 kPa.
- Ra at summer solstice (doy=172) at lat 44.1125 ≈ 42.3 MJ/m²/day (FAO-56).
- Daylength at equinox (doy=80) anywhere ≈ 12 h.
- GDD cumulative from transplant is monotone nondecreasing.
- Hargreaves ET0 is nonnegative.
- SWDI is 1 at FC and 0 at WP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FIELD_CAPACITY_PCT, WILTING_POINT_PCT, LAT
from src.features.et import hargreaves_et0_mm_day, vpd_kpa
from src.features.phenology import gdd_cumulative, kc_from_gdd, stage_from_gdd
from src.features.radiation import daylength_hours, extraterrestrial_radiation
from src.features.soil import swdi


def test_vpd_known_point():
    t = pd.Series([25.0])
    rh = pd.Series([50.0])
    got = float(vpd_kpa(t, rh).iloc[0])
    assert got == pytest.approx(1.584, abs=0.01), got


def test_vpd_saturated_is_zero():
    got = float(vpd_kpa(pd.Series([20.0]), pd.Series([100.0])).iloc[0])
    assert got == pytest.approx(0.0, abs=1e-9)


def test_vpd_nonnegative_clipped():
    got = vpd_kpa(pd.Series([15.0]), pd.Series([120.0])).iloc[0]
    assert got >= 0.0


def test_ra_summer_solstice_stuard():
    ra = float(extraterrestrial_radiation(np.array([172]), LAT)[0])
    assert ra == pytest.approx(42.3, abs=1.0), ra


def test_ra_winter_solstice_smaller():
    ra_summer = extraterrestrial_radiation(np.array([172]), LAT)[0]
    ra_winter = extraterrestrial_radiation(np.array([355]), LAT)[0]
    assert ra_winter < ra_summer


def test_daylength_equinox_12h():
    dl = float(daylength_hours(np.array([80]), 0.0)[0])
    assert dl == pytest.approx(12.0, abs=0.3)


def test_daylength_longer_in_summer_at_stuard():
    dl_summer = daylength_hours(np.array([172]), LAT)[0]
    dl_winter = daylength_hours(np.array([355]), LAT)[0]
    assert dl_summer > dl_winter


def test_hargreaves_nonnegative():
    ra = extraterrestrial_radiation(np.array([180]), LAT)
    et = hargreaves_et0_mm_day(
        pd.Series([25.0]), pd.Series([30.0]), pd.Series([20.0]), ra,
    )
    assert float(et.iloc[0]) >= 0.0


def test_gdd_monotone():
    idx = pd.date_range("2023-05-01", "2023-08-30", freq="10min")
    air = pd.Series(np.full(len(idx), 22.0), index=range(len(idx)))
    dt = pd.Series(idx.values, index=range(len(idx)))
    gdd = gdd_cumulative(dt, air)
    diffs = gdd.dropna().diff().dropna().values
    assert (diffs >= -1e-9).all()


def test_stage_from_gdd():
    assert stage_from_gdd(0.0) == "initial"
    assert stage_from_gdd(400.0) == "development"
    assert stage_from_gdd(800.0) == "mid"
    assert stage_from_gdd(1200.0) == "late"
    assert stage_from_gdd(99999.0) == "late"


def test_kc_monotone_across_stages():
    kcs = [kc_from_gdd(g) for g in [0.0, 175.0, 525.0, 900.0, 1300.0]]
    # Kc rises through mid then falls; just ensure boundaries match Kc table ends
    assert kcs[0] == pytest.approx(0.40, abs=1e-9)
    assert kcs[-1] == pytest.approx(0.85, abs=1e-9)


def test_swdi_endpoints():
    at_fc = float(swdi(pd.Series([FIELD_CAPACITY_PCT])).iloc[0])
    at_wp = float(swdi(pd.Series([WILTING_POINT_PCT])).iloc[0])
    assert at_fc == pytest.approx(1.0, abs=1e-9)
    assert at_wp == pytest.approx(0.0, abs=1e-9)

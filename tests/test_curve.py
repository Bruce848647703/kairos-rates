"""日计数、曲线与 bootstrap 测试。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from kairos_rates.curve import DiscountCurve
from kairos_rates.data import synthetic_curve
from kairos_rates.daycount import add_months, schedule, year_fraction


# --------------------------------------------------------------------------- #
# 日计数
# --------------------------------------------------------------------------- #
def test_year_fraction_conventions():
    d0, d1 = date(2024, 1, 1), date(2024, 7, 1)
    assert year_fraction(d0, d1, "ACT/365F") == pytest.approx(182 / 365)
    assert year_fraction(d0, d1, "ACT/360") == pytest.approx(182 / 360)
    assert year_fraction(date(2024, 1, 31), date(2024, 7, 31), "30/360") == pytest.approx(0.5)
    assert 0.49 < year_fraction(d0, d1, "ACT/ACT") < 0.51
    with pytest.raises(ValueError):
        year_fraction(d1, d0)


def test_add_months_handles_month_end():
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2023, 8, 31), 1) == date(2023, 9, 30)
    assert add_months(date(2024, 12, 15), 1) == date(2025, 1, 15)


def test_schedule_semi_annual():
    dates = schedule(date(2024, 1, 15), date(2026, 1, 15), 2)
    assert dates[0] == date(2024, 1, 15)
    assert dates[-1] == date(2026, 1, 15)
    assert len(dates) == 5
    with pytest.raises(ValueError):
        schedule(date(2024, 1, 1), date(2025, 1, 1), 5)


# --------------------------------------------------------------------------- #
# 曲线
# --------------------------------------------------------------------------- #
def test_curve_from_zero_roundtrip():
    times = [0.5, 1, 2, 5, 10, 30]
    zeros = [0.02, 0.021, 0.023, 0.026, 0.028, 0.03]
    curve = DiscountCurve.from_zero(times, zeros)
    assert np.allclose(curve.zero(times), zeros, atol=1e-14)
    assert np.allclose(curve.df(times), np.exp(-np.array(zeros) * np.array(times)))


def test_curve_df_is_positive_and_decreasing():
    curve = synthetic_curve()
    ts = np.linspace(curve.times[0], 30, 100)     # 首个 pillar 之前是平坦外推
    dfs = curve.df(ts)
    assert np.all(dfs > 0)
    assert np.all(np.diff(dfs) < 0)


def test_forward_matches_instantaneous_forward():
    curve = synthetic_curve()
    for t in (1.0, 5.0, 10.0):
        fwd = curve.forward(t, t + 1e-3)
        inst = curve.inst_forward(t + 5e-4)
        assert fwd == pytest.approx(inst, abs=2e-4)


def test_par_rate_identity():
    curve = synthetic_curve()
    for maturity in (2.0, 5.0, 10.0):
        par = curve.par_rate(maturity, 2)
        annuity = curve.annuity(maturity, 2)
        assert par * annuity + float(curve.df(maturity)) == pytest.approx(1.0)


def test_shifted_curve():
    curve = synthetic_curve()
    up = curve.shifted(100.0)                     # +100bp
    for t in curve.pillars:
        assert float(up.zero(t)) == pytest.approx(float(curve.zero(t)) + 0.01, abs=1e-12)


def test_key_bump_is_local():
    curve = synthetic_curve()
    bumped = curve.key_bumped(2, 100.0)
    assert float(bumped.zero(curve.pillars[2])) > float(curve.zero(curve.pillars[2]))
    assert float(bumped.df(curve.pillars[0])) == pytest.approx(float(curve.df(curve.pillars[0])))


def test_curve_validation():
    with pytest.raises(ValueError):
        DiscountCurve([1.0], [0.9])               # 点太少
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 2.0], [0.9])          # 长度不一致
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 1.0], [0.9, 0.8])     # 重复 pillar
    with pytest.raises(ValueError):
        DiscountCurve([0.0, 1.0], [1.0, 0.9])     # 非正时间
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 2.0], [0.9, 1.1])     # df > 1

    curve = DiscountCurve([2.0, 1.0], [0.8, 0.9])  # 乱序输入自动排序
    assert curve.pillars == [1.0, 2.0]
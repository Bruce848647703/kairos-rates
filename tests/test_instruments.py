"""债券、互换与关键利率久期测试。"""
from __future__ import annotations

import math

import pytest

from kairos_rates.curve import bootstrap
from kairos_rates.data import synthetic_curve, synthetic_quotes
from kairos_rates.instruments import (
    FixedRateBond,
    InterestRateSwap,
    key_rate_durations,
    parallel_dv01,
)


@pytest.fixture(scope="module")
def curve():
    return bootstrap(synthetic_quotes(synthetic_curve()))


# --------------------------------------------------------------------------- #
# 债券
# --------------------------------------------------------------------------- #
def test_bond_ytm_roundtrip(curve):
    bond = FixedRateBond(face=100.0, coupon_rate=0.03, maturity=10.0, frequency=2)
    price = bond.price(curve)
    y = bond.ytm(price)
    pv = sum(cf["amount"] * math.exp(-y * cf["time"]) for cf in bond.cashflows())
    assert pv == pytest.approx(price, rel=1e-9)


def test_zero_coupon_duration_equals_maturity(curve):
    bond = FixedRateBond(face=100.0, coupon_rate=0.0, maturity=10.0, frequency=2)
    assert bond.macaulay_duration(curve) == pytest.approx(10.0)
    assert bond.modified_duration(curve) == pytest.approx(10.0)
    assert bond.convexity(curve) == pytest.approx(100.0)


def test_convexity_and_duration_positive(curve):
    bond = FixedRateBond(100.0, 0.03, 30.0, 2)
    assert bond.macaulay_duration(curve) > 0
    assert bond.convexity(curve) > 0
    long_bond = FixedRateBond(100.0, 0.03, 30.0, 2).macaulay_duration(curve)
    short_bond = FixedRateBond(100.0, 0.03, 2.0, 2).macaulay_duration(curve)
    assert long_bond > short_bond


def test_bond_price_falls_when_curve_rises(curve):
    bond = FixedRateBond(100.0, 0.03, 10.0, 2)
    assert bond.price(curve.shifted(100.0)) < bond.price(curve)
    assert bond.price(curve.shifted(-100.0)) > bond.price(curve)


def test_accrued_interest(curve):
    bond = FixedRateBond(100.0, 0.04, 10.0, 2)
    coupon = 100.0 * 0.04 / 2
    assert bond.accrued_interest(0.25) == pytest.approx(coupon * 0.5)
    assert bond.accrued_interest(0.0) == 0.0
    assert bond.accrued_interest(10.0) == 0.0


def test_bond_validation():
    with pytest.raises(ValueError):
        FixedRateBond(face=0)
    with pytest.raises(ValueError):
        FixedRateBond(maturity=0.0, issue=1.0)


# --------------------------------------------------------------------------- #
# 互换
# --------------------------------------------------------------------------- #
def test_par_swap_has_zero_pv(curve):
    par = curve.par_rate(5.0, 2)
    swap = InterestRateSwap(1_000_000.0, par, 5.0, 2)
    assert swap.pv(curve) == pytest.approx(0.0, abs=1e-6)
    assert swap.par_rate(curve) == pytest.approx(par, rel=1e-12)


def test_swap_dv01_signs(curve):
    par = curve.par_rate(5.0, 2)
    payer = InterestRateSwap(1_000_000.0, par, 5.0, 2, payer=True)
    receiver = InterestRateSwap(1_000_000.0, par, 5.0, 2, payer=False)
    assert payer.dv01(curve) > 0
    assert receiver.dv01(curve) == pytest.approx(-payer.dv01(curve), rel=1e-10)


def test_swap_value_moves_with_fixed_rate(curve):
    par = curve.par_rate(5.0, 2)
    cheap = InterestRateSwap(1_000_000.0, par - 0.01, 5.0, 2, payer=True)
    assert cheap.pv(curve) > 0
    rich = InterestRateSwap(1_000_000.0, par + 0.01, 5.0, 2, payer=True)
    assert rich.pv(curve) < 0


def test_swap_validation():
    with pytest.raises(ValueError):
        InterestRateSwap(notional=0)
    with pytest.raises(ValueError):
        InterestRateSwap(maturity=0)


# --------------------------------------------------------------------------- #
# 风险
# --------------------------------------------------------------------------- #
def test_key_rate_durations_capture_parallel(curve):
    bond = FixedRateBond(100.0, 0.03, 10.0, 2)
    report = key_rate_durations(curve, bond.price, bp=1.0)
    assert len(report["rows"]) == len(curve.times)
    assert report["parallel_dv01"] != pytest.approx(0.0)
    assert 0.9 <= report["capture_ratio"] <= 1.1
    # 10Y 债券对 20Y/30Y pillar 几乎不敏感
    long_pillars = [r for r in report["rows"] if r["pillar"] > 10.5]
    assert all(abs(r["dv01"]) < 1e-6 for r in long_pillars)


def test_parallel_dv01_matches_finite_difference(curve):
    bond = FixedRateBond(100.0, 0.03, 10.0, 2)
    dv01 = parallel_dv01(curve, bond.price, bp=1.0)
    expected = bond.price(curve.shifted(1.0)) - bond.price(curve)
    assert dv01 == pytest.approx(expected, rel=1e-12)
    assert dv01 < 0                            # 利率上行，债券价格下跌
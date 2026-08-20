"""参数化曲线与短期利率模型测试。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kairos_rates.curve import DiscountCurve
from kairos_rates.data import synthetic_curve
from kairos_rates.models import (
    CIR,
    HullWhite,
    NelsonSiegel,
    Svensson,
    Vasicek,
    fit_nelson_siegel,
    fit_svensson,
)

PILLARS = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype="float64")


# --------------------------------------------------------------------------- #
# Nelson-Siegel / Svensson
# --------------------------------------------------------------------------- #
def test_nelson_siegel_recovers_parameters():
    true = NelsonSiegel(0.030, -0.015, 0.010, 2.0)
    fit, rmse = fit_nelson_siegel(PILLARS, true.zero(PILLARS))
    assert rmse < 1e-6
    assert fit.beta0 == pytest.approx(0.030, abs=1e-5)
    assert fit.beta1 == pytest.approx(-0.015, abs=1e-5)
    assert fit.beta2 == pytest.approx(0.010, abs=1e-5)
    assert fit.tau == pytest.approx(2.0, abs=1e-2)


def test_nelson_siegel_short_end_limit():
    model = NelsonSiegel(0.03, -0.02, 0.01, 2.0)
    assert float(model.zero(1e-8)) == pytest.approx(0.03 - 0.02, abs=1e-6)
    assert float(model.zero(1e7)) == pytest.approx(0.03, abs=1e-6)


def test_svensson_recovers_parameters():
    true = Svensson(0.030, -0.020, 0.015, -0.010, 1.5, 6.0)
    fit, rmse = fit_svensson(PILLARS, true.zero(PILLARS))
    assert rmse < 1e-6
    assert fit.beta0 == pytest.approx(0.030, abs=1e-4)


def test_fit_requires_enough_points():
    with pytest.raises(ValueError):
        fit_nelson_siegel([1, 2, 3], [0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        fit_svensson([1, 2, 3, 4, 5], [0.01] * 5)


def test_model_validation():
    with pytest.raises(ValueError):
        NelsonSiegel(0.03, -0.01, 0.01, 0.0)
    with pytest.raises(ValueError):
        Svensson(0.03, -0.01, 0.01, 0.0, 1.0, 1.0)


# --------------------------------------------------------------------------- #
# Vasicek
# --------------------------------------------------------------------------- #
def test_vasicek_bond_price_properties():
    model = Vasicek(kappa=0.8, theta=0.035, sigma=0.01, r0=0.02)
    prices = [model.bond_price(t) for t in (1, 5, 10, 30)]
    assert all(0 < p <= 1 for p in prices)
    assert all(b < a for a, b in zip(prices, prices[1:]))
    assert model.bond_price(0.0) == 1.0
    assert model.mean(0.0) == pytest.approx(0.02)
    assert model.variance(0.0) == pytest.approx(0.0)


def test_vasicek_mean_reversion():
    model = Vasicek(kappa=2.0, theta=0.04, sigma=0.01, r0=0.01)
    paths = model.simulate(steps=252, maturity=2.0, n_paths=20000, seed=3)
    assert paths.shape == (20000, 253)
    terminal = paths[:, -1]
    assert abs(terminal.mean() - model.mean(2.0)) < 5e-4
    assert terminal.std() == pytest.approx(math.sqrt(model.variance(2.0)), rel=0.05)


def test_vasicek_bond_price_matches_mc_discounting():
    model = Vasicek(kappa=0.5, theta=0.03, sigma=0.008, r0=0.025)
    maturity = 1.0
    steps = 252
    paths = model.simulate(steps=steps, maturity=maturity, n_paths=20000, seed=11)
    dt = maturity / steps
    integral = paths[:, :-1].sum(axis=1) * dt          # 左端点积分
    mc = float(np.mean(np.exp(-integral)))
    assert mc == pytest.approx(model.bond_price(maturity), rel=0.01)


# --------------------------------------------------------------------------- #
# CIR
# --------------------------------------------------------------------------- #
def test_cir_properties():
    model = CIR(kappa=1.2, theta=0.03, sigma=0.06, r0=0.02)
    assert model.feller_ok
    paths = model.simulate(steps=200, maturity=2.0, n_paths=500, seed=5)
    assert np.all(paths >= 0)
    assert model.bond_price(10.0) < model.bond_price(1.0)
    assert model.bond_price(0.0) == 1.0


def test_cir_zero_vol_limit():
    model = CIR(kappa=1.0, theta=0.03, sigma=0.0, r0=0.03)
    assert model.zero_rate(5.0) == pytest.approx(0.03, abs=1e-3)


# --------------------------------------------------------------------------- #
# Hull-White
# --------------------------------------------------------------------------- #
def test_hull_white_reprices_initial_curve():
    curve = synthetic_curve()
    model = HullWhite(a=0.1, sigma=0.01, curve=curve)
    for t in (0.5, 1, 2, 5, 10, 20, 30):
        assert model.bond_price(t) == pytest.approx(float(curve.df(t)), rel=1e-12)
        assert model.zero_rate(t) == pytest.approx(float(curve.zero(t)), rel=1e-12)


def test_hull_white_forward_prices_are_positive():
    curve = synthetic_curve()
    model = HullWhite(a=0.2, sigma=0.012, curve=curve)
    for t, T in ((1, 5), (5, 10), (10, 30)):
        price = model.bond_price(T, t=t)
        assert 0 < price < 1
    assert model.simulate(steps=60, maturity=1.0, n_paths=20, seed=1).shape == (20, 61)


def test_hull_white_validation():
    curve = synthetic_curve()
    with pytest.raises(ValueError):
        HullWhite(a=0.0, sigma=0.01, curve=curve)
    with pytest.raises(ValueError):
        HullWhite(a=0.1, sigma=-0.01, curve=curve)
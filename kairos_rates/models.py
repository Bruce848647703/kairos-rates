"""曲线拟合与短期利率模型。

- **Nelson-Siegel / Svensson**：参数化收益率曲线，拟合快、可外推；
- **Vasicek / CIR**：经典短期利率模型，有解析债券价格；
- **Hull-White 1F**：``dr = (θ(t) - a·r)dt + σ dW``，θ(t) 校准到初始曲线，
  因此**在 t=0 精确重定价整条曲线**，是利率衍生品定价的标准模型。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .curve import DiscountCurve
from .numerics import nelder_mead, weighted_sse


# --------------------------------------------------------------------------- #
# Nelson-Siegel / Svensson
# --------------------------------------------------------------------------- #
def _ns_basis(t: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(t, dtype="float64") / float(tau)
    safe = np.where(x < 1e-10, 1.0, x)
    decay = (1.0 - np.exp(-x)) / safe
    decay = np.where(x < 1e-10, 1.0, decay)
    return np.column_stack([np.ones_like(x), decay, decay - np.exp(-x)])


@dataclass
class NelsonSiegel:
    """``y(t) = β0 + β1·f1 + β2·f2``，τ 控制衰减速度。"""

    beta0: float
    beta1: float
    beta2: float
    tau: float

    def __post_init__(self) -> None:
        if not self.tau > 0:
            raise ValueError("tau 必须为正")

    def zero(self, t):
        t_arr = np.asarray(t, dtype="float64")
        basis = _ns_basis(t_arr.ravel(), self.tau)
        values = basis @ np.array([self.beta0, self.beta1, self.beta2])
        return float(values[0]) if t_arr.ndim == 0 else values.reshape(t_arr.shape)

    def to_dict(self) -> Dict[str, float]:
        return {"beta0": self.beta0, "beta1": self.beta1,
                "beta2": self.beta2, "tau": self.tau}


@dataclass
class Svensson:
    """Nelson-Siegel + 第二个衰减项（共 6 参数）。"""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def __post_init__(self) -> None:
        if not self.tau1 > 0 or not self.tau2 > 0:
            raise ValueError("tau1 / tau2 必须为正")
        if abs(self.tau1 - self.tau2) < 1e-12:
            raise ValueError("tau1 与 tau2 不能相同")

    def zero(self, t):
        t_arr = np.asarray(t, dtype="float64")
        flat = t_arr.ravel()
        b1 = _ns_basis(flat, self.tau1)
        x2 = flat / self.tau2
        safe2 = np.where(x2 < 1e-10, 1.0, x2)
        decay2 = np.where(x2 < 1e-10, 1.0, (1.0 - np.exp(-x2)) / safe2)
        third = decay2 - np.exp(-x2)
        values = (b1 @ np.array([self.beta0, self.beta1, self.beta2])
                  + self.beta3 * third)
        return float(values[0]) if t_arr.ndim == 0 else values.reshape(t_arr.shape)

    def to_dict(self) -> Dict[str, float]:
        return {"beta0": self.beta0, "beta1": self.beta1, "beta2": self.beta2,
                "beta3": self.beta3, "tau1": self.tau1, "tau2": self.tau2}


def _fit_betas(design: np.ndarray, rates: np.ndarray,
               weights: Optional[np.ndarray] = None) -> np.ndarray:
    if weights is None:
        coef, *_ = np.linalg.lstsq(design, rates, rcond=None)
    else:
        sw = np.sqrt(weights)
        coef, *_ = np.linalg.lstsq(design * sw[:, None], rates * sw, rcond=None)
    return coef


def fit_nelson_siegel(times: Sequence[float], rates: Sequence[float],
                      weights: Optional[Sequence[float]] = None) -> Tuple[NelsonSiegel, float]:
    """网格搜索 τ + 线性最小二乘求 β，返回 ``(模型, RMSE)``。"""
    t = np.asarray(times, dtype="float64").ravel()
    y = np.asarray(rates, dtype="float64").ravel()
    if t.size != y.size or t.size < 4:
        raise ValueError("至少需要 4 个点")
    w = None if weights is None else np.asarray(weights, dtype="float64").ravel()

    best = None
    best_sse = np.inf
    for tau in np.geomspace(0.05, 20.0, 60):
        coef = _fit_betas(_ns_basis(t, tau), y, w)
        model = NelsonSiegel(float(coef[0]), float(coef[1]), float(coef[2]), float(tau))
        sse = weighted_sse(np.asarray(model.zero(t)), y, w)
        if sse < best_sse:
            best_sse, best = sse, model
    assert best is not None

    def objective(x: np.ndarray) -> float:
        tau = float(np.exp(x[0]))
        coef = _fit_betas(_ns_basis(t, tau), y, w)
        return weighted_sse(np.asarray(_ns_basis(t, tau) @ coef), y, w)

    x_best, _ = nelder_mead(objective, np.array([np.log(best.tau)]), step=0.05, max_iter=400)
    tau = float(np.exp(x_best[0]))
    coef = _fit_betas(_ns_basis(t, tau), y, w)
    model = NelsonSiegel(float(coef[0]), float(coef[1]), float(coef[2]), tau)
    rmse = float(np.sqrt(np.mean((np.asarray(model.zero(t)) - y) ** 2)))
    return model, rmse


def fit_svensson(times: Sequence[float], rates: Sequence[float],
                 weights: Optional[Sequence[float]] = None) -> Tuple[Svensson, float]:
    """粗网格 (τ1, τ2) + 线性最小二乘求 4 个 β，再用 Nelder-Mead 精修。"""
    t = np.asarray(times, dtype="float64").ravel()
    y = np.asarray(rates, dtype="float64").ravel()
    if t.size != y.size or t.size < 6:
        raise ValueError("Svensson 至少需要 6 个点")
    w = None if weights is None else np.asarray(weights, dtype="float64").ravel()

    def design(tau1: float, tau2: float) -> np.ndarray:
        base = _ns_basis(t, tau1)
        x2 = t / tau2
        safe = np.where(x2 < 1e-10, 1.0, x2)
        decay = np.where(x2 < 1e-10, 1.0, (1.0 - np.exp(-x2)) / safe)
        return np.column_stack([base, decay - np.exp(-x2)])

    best = None
    best_sse = np.inf
    for tau1 in np.geomspace(0.1, 15.0, 12):
        for tau2 in np.geomspace(0.1, 25.0, 12):
            if abs(tau1 - tau2) < 0.05:
                continue
            coef = _fit_betas(design(float(tau1), float(tau2)), y, w)
            pred = design(float(tau1), float(tau2)) @ coef
            sse = weighted_sse(pred, y, w)
            if sse < best_sse:
                best_sse, best = sse, (float(tau1), float(tau2), coef)
    assert best is not None
    tau1, tau2, coef = best

    def objective(x: np.ndarray) -> float:
        a, b = float(np.exp(x[0])), float(np.exp(x[1]))
        if abs(a - b) < 1e-6:
            return 1e12
        c = _fit_betas(design(a, b), y, w)
        return weighted_sse(design(a, b) @ c, y, w)

    x_best, _ = nelder_mead(objective, np.array([np.log(tau1), np.log(tau2)]),
                            step=0.05, max_iter=600)
    tau1 = float(np.exp(x_best[0]))
    tau2 = float(np.exp(x_best[1]))
    if abs(tau1 - tau2) < 1e-6:
        tau2 = tau1 * 1.5
    coef = _fit_betas(design(tau1, tau2), y, w)
    model = Svensson(float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3]),
                     tau1, tau2)
    rmse = float(np.sqrt(np.mean((np.asarray(model.zero(t)) - y) ** 2)))
    return model, rmse


# --------------------------------------------------------------------------- #
# 短期利率模型
# --------------------------------------------------------------------------- #
def _validate_model(kappa: float, theta: float, sigma: float) -> None:
    if not kappa > 0:
        raise ValueError("kappa 必须为正")
    if not np.isfinite(theta):
        raise ValueError("theta 必须有限")
    if not sigma >= 0:
        raise ValueError("sigma 不能为负")


@dataclass
class Vasicek:
    """``dr = κ(θ - r)dt + σ dW``；债券价格解析：``P(t,T)=A·exp(-B·r_t)``。"""

    kappa: float
    theta: float
    sigma: float
    r0: float = 0.03

    def __post_init__(self) -> None:
        _validate_model(self.kappa, self.theta, self.sigma)
        if not np.isfinite(self.r0):
            raise ValueError("r0 必须有限")

    def _AB(self, tau: float) -> Tuple[float, float]:
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        b = (1.0 - math.exp(-kappa * tau)) / kappa
        a = math.exp((theta - sigma ** 2 / (2.0 * kappa ** 2)) * (b - tau)
                     - sigma ** 2 * b ** 2 / (4.0 * kappa))
        return a, b

    def bond_price(self, maturity: float, t: float = 0.0,
                   r: Optional[float] = None) -> float:
        tau = float(maturity) - float(t)
        if tau <= 0:
            return 1.0
        a, b = self._AB(tau)
        rate = self.r0 if r is None else float(r)
        return float(a * math.exp(-b * rate))

    def zero_rate(self, maturity: float, t: float = 0.0,
                  r: Optional[float] = None) -> float:
        tau = float(maturity) - float(t)
        if tau <= 0:
            return 0.0
        return -math.log(self.bond_price(maturity, t, r)) / tau

    def mean(self, t: float) -> float:
        """``E[r_t]``。"""
        return self.theta + (self.r0 - self.theta) * math.exp(-self.kappa * t)

    def variance(self, t: float) -> float:
        return self.sigma ** 2 * (1.0 - math.exp(-2.0 * self.kappa * t)) / (2.0 * self.kappa)

    def simulate(self, steps: int = 252, maturity: float = 1.0,
                 n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """精确离散化模拟，返回 ``(n_paths, steps+1)`` 的短期利率路径。"""
        steps, n_paths = int(steps), int(n_paths)
        if steps < 1 or n_paths < 1:
            raise ValueError("steps / n_paths 必须 >= 1")
        dt = float(maturity) / steps
        rng = np.random.default_rng(seed)
        decay = math.exp(-self.kappa * dt)
        shock = self.sigma * math.sqrt((1.0 - decay ** 2) / (2.0 * self.kappa))
        paths = np.empty((n_paths, steps + 1))
        paths[:, 0] = self.r0
        for i in range(1, steps + 1):
            paths[:, i] = (self.theta + (paths[:, i - 1] - self.theta) * decay
                           + shock * rng.standard_normal(n_paths))
        return paths


@dataclass
class CIR:
    """``dr = κ(θ - r)dt + σ√r dW``；债券价格解析，利率非负（Feller 条件满足时）。"""

    kappa: float
    theta: float
    sigma: float
    r0: float = 0.03

    def __post_init__(self) -> None:
        _validate_model(self.kappa, self.theta, self.sigma)
        if not self.r0 >= 0:
            raise ValueError("r0 不能为负")

    @property
    def feller_ok(self) -> bool:
        return 2.0 * self.kappa * self.theta >= self.sigma ** 2

    def _AB(self, tau: float) -> Tuple[float, float]:
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        if sigma == 0:
            b = (1.0 - math.exp(-kappa * tau)) / kappa
            return math.exp(theta * (b - tau) - 0.0 * theta * b), b
        h = math.sqrt(kappa ** 2 + 2.0 * sigma ** 2)
        exp_h = math.exp(h * tau)
        denominator = (h + kappa) * (exp_h - 1.0) + 2.0 * h
        b = 2.0 * (exp_h - 1.0) / denominator
        a = ((2.0 * h * math.exp((kappa + h) * tau / 2.0) / denominator)
             ** (2.0 * kappa * theta / sigma ** 2))
        return a, b

    def bond_price(self, maturity: float, t: float = 0.0,
                   r: Optional[float] = None) -> float:
        tau = float(maturity) - float(t)
        if tau <= 0:
            return 1.0
        a, b = self._AB(tau)
        rate = self.r0 if r is None else max(float(r), 0.0)
        return float(a * math.exp(-b * rate))

    def zero_rate(self, maturity: float, t: float = 0.0,
                  r: Optional[float] = None) -> float:
        tau = float(maturity) - float(t)
        if tau <= 0:
            return 0.0
        return -math.log(self.bond_price(maturity, t, r)) / tau

    def simulate(self, steps: int = 252, maturity: float = 1.0,
                 n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
        steps, n_paths = int(steps), int(n_paths)
        if steps < 1 or n_paths < 1:
            raise ValueError("steps / n_paths 必须 >= 1")
        dt = float(maturity) / steps
        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, steps + 1))
        paths[:, 0] = self.r0
        for i in range(1, steps + 1):
            prev = np.maximum(paths[:, i - 1], 0.0)
            paths[:, i] = np.maximum(
                prev + self.kappa * (self.theta - prev) * dt
                + self.sigma * np.sqrt(prev * dt) * rng.standard_normal(n_paths), 0.0)
        return paths


@dataclass
class HullWhite:
    """单因子 Hull-White：``dr = (θ(t) - a·r)dt + σ dW``，θ(t) 匹配初始曲线。"""

    a: float
    sigma: float
    curve: DiscountCurve
    r0: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.a > 0:
            raise ValueError("a 必须为正")
        if not self.sigma >= 0:
            raise ValueError("sigma 不能为负")
        if self.r0 is None:
            self.r0 = self.curve.inst_forward(1e-4)

    def _B(self, tau: float) -> float:
        return (1.0 - math.exp(-self.a * tau)) / self.a

    def bond_price(self, maturity: float, t: float = 0.0,
                   r: Optional[float] = None) -> float:
        """``P(t,T)=A(t,T)·exp(-B(t,T)·r_t)``，在 t=0 精确重定价初始曲线。"""
        maturity, t = float(maturity), float(t)
        tau = maturity - t
        if tau <= 0:
            return 1.0
        if t <= 1e-12:
            return float(self.curve.df(maturity))
        b = self._B(tau)
        p0t = float(self.curve.df(t))
        p0T = float(self.curve.df(maturity))
        forward = self.curve.inst_forward(t)
        a_t = (p0T / p0t) * math.exp(b * forward
                                     - self.sigma ** 2 / (4.0 * self.a)
                                     * (1.0 - math.exp(-2.0 * self.a * t)) * b * b)
        rate = float(self.r0) if r is None else float(r)
        return float(a_t * math.exp(-b * rate))

    def zero_rate(self, maturity: float, t: float = 0.0,
                  r: Optional[float] = None) -> float:
        tau = float(maturity) - float(t)
        if tau <= 0:
            return 0.0
        return -math.log(self.bond_price(maturity, t, r)) / tau

    def theta(self, t: float, h: float = 1e-4) -> float:
        """风险中性漂移里的 θ(t)，用于模拟。"""
        f0 = self.curve.inst_forward(t)
        f1 = self.curve.inst_forward(t + h)
        derivative = (f1 - f0) / h
        return (derivative + self.a * f0
                + self.sigma ** 2 / (2.0 * self.a) * (1.0 - math.exp(-2.0 * self.a * t)))

    def simulate(self, steps: int = 252, maturity: float = 1.0,
                 n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
        steps, n_paths = int(steps), int(n_paths)
        if steps < 1 or n_paths < 1:
            raise ValueError("steps / n_paths 必须 >= 1")
        dt = float(maturity) / steps
        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, steps + 1))
        paths[:, 0] = float(self.r0)
        for i in range(1, steps + 1):
            t = (i - 1) * dt
            theta = self.theta(t) if t > 0 else self.theta(1e-6)
            paths[:, i] = (paths[:, i - 1]
                           + (theta - self.a * paths[:, i - 1]) * dt
                           + self.sigma * math.sqrt(dt) * rng.standard_normal(n_paths))
        return paths


__all__ = [
    "CIR",
    "HullWhite",
    "NelsonSiegel",
    "Svensson",
    "Vasicek",
    "fit_nelson_siegel",
    "fit_svensson",
]
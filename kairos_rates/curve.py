"""贴现曲线与 bootstrap。

约定
----
- 时间单位「年」，利率**连续复利**：``df(t) = exp(-z(t)·t)``；
- 期限之间对 ``ln df`` 线性插值（等价于分段常数瞬时远期），端点平坦外推；
- 曲线只保存 (t, df) 两组数，任何利率都能从它们推出来。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[float, Sequence[float]]


class DiscountCurve:
    """贴现曲线。"""

    def __init__(self, times: Sequence[float], discount_factors: Sequence[float]) -> None:
        t = np.asarray(times, dtype="float64").ravel()
        d = np.asarray(discount_factors, dtype="float64").ravel()
        if t.size != d.size:
            raise ValueError("times 与 discount_factors 长度必须一致")
        if t.size < 2:
            raise ValueError("至少需要 2 个曲线点")
        if np.any(t <= 0):
            raise ValueError("时间必须为正")
        if np.any(d <= 0) or np.any(d > 1.0 + 1e-9):
            raise ValueError("贴现因子必须在 (0, 1] 内")
        order = np.argsort(t)
        self.times = t[order]
        self.dfs = d[order]
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("时间必须严格递增（不要重复 pillar）")
        self.log_dfs = np.log(self.dfs)

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_zero(cls, times: Sequence[float], zero_rates: Sequence[float]) -> "DiscountCurve":
        t = np.asarray(times, dtype="float64").ravel()
        z = np.asarray(zero_rates, dtype="float64").ravel()
        if t.size != z.size:
            raise ValueError("times 与 zero_rates 长度必须一致")
        return cls(t, np.exp(-z * t))

    @classmethod
    def flat(cls, rate: float, max_maturity: float = 30.0) -> "DiscountCurve":
        return cls.from_zero([0.5, max_maturity], [float(rate), float(rate)])

    # ------------------------------------------------------------------ 查询
    def df(self, t: ArrayLike):
        """贴现因子 ``df(t)``（支持数组）。"""
        arr = np.asarray(t, dtype="float64")
        if arr.ndim == 0:
            value = float(arr)
            if value <= 0:
                return 1.0
            return float(np.exp(np.interp(value, self.times, self.log_dfs)))
        out = np.exp(np.interp(arr, self.times, self.log_dfs))
        out = np.where(arr <= 0, 1.0, out)
        return out

    def zero(self, t: ArrayLike):
        """连续复利零息利率 ``-ln df(t) / t``。"""
        arr = np.asarray(t, dtype="float64")
        out = -np.log(np.asarray(self.df(arr), dtype="float64")) / np.maximum(arr, 1e-12)
        if arr.ndim == 0:
            return float(out)
        return out

    def forward(self, start: float, end: float, compounding: str = "continuous") -> float:
        """``[start, end]`` 的远期利率。"""
        if end <= start:
            raise ValueError("end 必须大于 start")
        ratio = float(self.df(start)) / float(self.df(end))
        tau = end - start
        if compounding == "continuous":
            return float(np.log(ratio) / tau)
        if compounding == "simple":
            return (ratio - 1.0) / tau
        if compounding == "annual":
            return ratio ** (1.0 / tau) - 1.0
        raise ValueError("compounding 必须是 continuous / simple / annual")

    def inst_forward(self, t: float, h: float = 1e-4) -> float:
        """瞬时远期利率 ``f(0,t)``（数值微分）。"""
        t = float(t)
        lo, hi = max(t - h, 1e-6), t + h
        ln_lo = float(np.log(np.asarray(self.df(lo))))
        ln_hi = float(np.log(np.asarray(self.df(hi))))
        return (ln_lo - ln_hi) / (hi - lo)

    def annuity(self, maturity: float, frequency: int = 2) -> float:
        """年金因子 ``Σ τ_i · df(T_i)``。"""
        n, tau = self._payment_grid(maturity, frequency)
        return float(sum(tau * float(self.df(i * tau)) for i in range(1, n + 1)))

    def par_rate(self, maturity: float, frequency: int = 2) -> float:
        """平价互换利率（单曲线口径）。"""
        n, tau = self._payment_grid(maturity, frequency)
        annuity = float(sum(tau * float(self.df(i * tau)) for i in range(1, n + 1)))
        return (1.0 - float(self.df(n * tau))) / annuity

    @staticmethod
    def _payment_grid(maturity: float, frequency: int):
        frequency = int(frequency)
        if frequency < 1:
            raise ValueError("frequency 必须 >= 1")
        tau = 1.0 / frequency
        n = max(1, int(round(float(maturity) * frequency)))
        return n, tau

    # ------------------------------------------------------------------ 变形
    def shifted(self, bp: float) -> "DiscountCurve":
        """零息利率平行平移 ``bp`` 个基点。"""
        zeros = np.asarray(self.zero(self.times), dtype="float64") + float(bp) / 10_000.0
        return DiscountCurve.from_zero(self.times, zeros)

    def key_bumped(self, pillar_index: int, bp: float) -> "DiscountCurve":
        """对单个 pillar 的贴现因子做 ``bp`` 个基点的局部冲击。"""
        dfs = self.dfs.copy()
        i = int(pillar_index)
        dfs[i] *= np.exp(-float(bp) / 10_000.0 * self.times[i])
        return DiscountCurve(self.times, dfs)

    # ------------------------------------------------------------------ 展示
    @property
    def pillars(self) -> List[float]:
        return [float(t) for t in self.times]

    def grid(self, n: int = 20) -> Dict[str, Any]:
        ts = np.linspace(self.times[0], self.times[-1], int(n))
        return {"times": ts.tolist(), "zero": np.asarray(self.zero(ts)).tolist(),
                "df": np.asarray(self.df(ts)).tolist()}

    def to_dict(self) -> Dict[str, Any]:
        return {"times": self.times.tolist(), "dfs": self.dfs.tolist(),
                "zeros": np.asarray(self.zero(self.times)).tolist()}


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
@dataclass
class CurveQuote:
    """一条曲线工具报价。``kind`` 支持 ``deposit`` / ``swap`` / ``zero``。"""

    kind: str
    maturity: float
    rate: float
    frequency: int = 2

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        if self.kind not in ("deposit", "swap", "zero"):
            raise ValueError("kind 必须是 deposit / swap / zero")
        if not float(self.maturity) > 0:
            raise ValueError("maturity 必须为正")
        if float(self.rate) <= -0.999:
            raise ValueError("rate 必须大于 -100%")
        self.maturity = float(self.maturity)
        self.rate = float(self.rate)
        self.frequency = int(self.frequency)


def bootstrap(quotes: Iterable[CurveQuote], *,
              tol: float = 1e-13, max_iter: int = 60) -> DiscountCurve:
    """逐点 bootstrap + Newton 联立精修。

    先用「顺序反解」得到初值（快但中间付息日会被平外推），再用 Newton 同时求解所有
    pillar，使**每条报价都能被曲线精确重定价**（残差 < ``tol``）。
    """
    items = sorted(list(quotes), key=lambda q: q.maturity)
    if not items:
        raise ValueError("至少需要一条报价")

    # pillar 对齐到付息日
    pillars: List[float] = []
    for quote in items:
        t = quote.maturity
        if quote.kind == "swap":
            n = max(1, int(round(t * quote.frequency)))
            t = n / quote.frequency
        if pillars and t <= pillars[-1] + 1e-10:
            raise ValueError(f"报价期限必须严格递增（重复或乱序：{t}）")
        pillars.append(float(t))

    # 1) 顺序反解作为初值
    seed_times: List[float] = []
    seed_dfs: List[float] = []

    def interp_df(t: float) -> float:
        if not seed_times:
            return 1.0
        return float(np.exp(np.interp(t, seed_times, np.log(seed_dfs))))

    for quote, t in zip(items, pillars):
        if quote.kind == "deposit":
            df = 1.0 / (1.0 + quote.rate * t)
        elif quote.kind == "zero":
            df = float(np.exp(-quote.rate * t))
        else:
            tau = 1.0 / quote.frequency
            n = max(1, int(round(t * quote.frequency)))
            accrued = sum(tau * interp_df(i * tau) for i in range(1, n))
            df = (1.0 - quote.rate * accrued) / (1.0 + quote.rate * tau)
        if not np.isfinite(df) or df <= 0:
            raise ValueError(f"bootstrap 在 {t:.4f}y 处得到非法贴现因子 {df!r}")
        seed_times.append(t)
        seed_dfs.append(float(df))

    times = np.asarray(seed_times, dtype="float64")
    x = np.log(np.asarray(seed_dfs, dtype="float64"))

    def curve_from(vec: np.ndarray) -> DiscountCurve:
        return DiscountCurve(times, np.exp(np.clip(vec, -60.0, 0.0)))

    def residual(vec: np.ndarray) -> np.ndarray:
        curve = curve_from(vec)
        out = []
        for quote, t in zip(items, pillars):
            if quote.kind == "deposit":
                target = 1.0 / (1.0 + quote.rate * t)
                out.append(float(curve.df(t)) - target)
            elif quote.kind == "zero":
                out.append(float(curve.zero(t)) - quote.rate)
            else:
                out.append(float(curve.par_rate(t, quote.frequency)) - quote.rate)
        return np.asarray(out, dtype="float64")

    # 2) Newton（数值 Jacobian + 阻尼线搜索）
    current = residual(x)
    for _ in range(int(max_iter)):
        if float(np.max(np.abs(current))) < tol:
            break
        n = x.size
        jac = np.zeros((n, n))
        h = 1e-7
        base = current.copy()
        for j in range(n):
            step = np.zeros(n)
            step[j] = h
            jac[:, j] = (residual(x + step) - base) / h
        try:
            delta = np.linalg.solve(jac, -current)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jac, -current, rcond=None)[0]

        scale = 1.0
        accepted = False
        for _ in range(20):
            candidate = x + scale * delta
            if np.all(np.isfinite(candidate)):
                trial = residual(candidate)
                if float(np.max(np.abs(trial))) < float(np.max(np.abs(current))):
                    x, current = candidate, trial
                    accepted = True
                    break
            scale *= 0.5
        if not accepted:
            break

    return curve_from(x)


__all__ = ["CurveQuote", "DiscountCurve", "bootstrap"]
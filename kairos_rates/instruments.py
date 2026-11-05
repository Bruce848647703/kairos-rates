"""利率工具：固定利率债券、利率互换、关键利率久期。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


from .curve import DiscountCurve


def _pv_at_yield(amounts: Sequence[float], times: Sequence[float], y: float) -> float:
    return float(sum(a * math.exp(-y * t) for a, t in zip(amounts, times)))


@dataclass
class FixedRateBond:
    """固定利率付息债券（按年化期限表示，连续复利贴现）。"""

    face: float = 100.0
    coupon_rate: float = 0.03
    maturity: float = 10.0
    frequency: int = 2
    issue: float = 0.0

    def __post_init__(self) -> None:
        if self.face <= 0:
            raise ValueError("face 必须为正")
        if self.frequency < 1:
            raise ValueError("frequency 必须 >= 1")
        if self.maturity <= self.issue:
            raise ValueError("maturity 必须晚于 issue")

    @property
    def periods(self) -> int:
        return max(1, int(round((self.maturity - self.issue) * self.frequency)))

    def payment_times(self) -> List[float]:
        tau = 1.0 / self.frequency
        return [self.issue + i * tau for i in range(1, self.periods + 1)]

    def cashflows(self) -> List[Dict[str, float]]:
        """每期现金流：``{"time": t, "amount": x}``。"""
        coupon = self.face * self.coupon_rate / self.frequency
        flows = [{"time": t, "amount": coupon} for t in self.payment_times()]
        flows[-1]["amount"] += self.face
        return flows

    def price(self, curve: DiscountCurve) -> float:
        return float(sum(cf["amount"] * float(curve.df(cf["time"]))
                         for cf in self.cashflows()))

    def ytm(self, price: float, tol: float = 1e-10) -> float:
        """连续复利到期收益率（二分法）。"""
        if price <= 0:
            raise ValueError("price 必须为正")
        flows = self.cashflows()
        times = [cf["time"] for cf in flows]
        amounts = [cf["amount"] for cf in flows]
        lo, hi = -0.99, 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            value = _pv_at_yield(amounts, times, mid)
            if abs(value - price) < tol:
                return mid
            if value > price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _pv_weights(self, curve: DiscountCurve):
        flows = self.cashflows()
        pvs = [(cf["time"], cf["amount"] * float(curve.df(cf["time"]))) for cf in flows]
        total = sum(pv for _, pv in pvs)
        if total <= 0:
            raise ValueError("债券现值非正，无法计算久期")
        return pvs, total

    def macaulay_duration(self, curve: DiscountCurve) -> float:
        pvs, total = self._pv_weights(curve)
        return float(sum(t * pv for t, pv in pvs) / total)

    def modified_duration(self, curve: DiscountCurve) -> float:
        """连续复利下修正久期 = 麦考利久期。"""
        return self.macaulay_duration(curve)

    def convexity(self, curve: DiscountCurve) -> float:
        pvs, total = self._pv_weights(curve)
        return float(sum(t * t * pv for t, pv in pvs) / total)

    def accrued_interest(self, settlement: float) -> float:
        """应计利息（线性）。"""
        s = float(settlement)
        if s <= self.issue or s >= self.maturity:
            return 0.0
        coupon = self.face * self.coupon_rate / self.frequency
        tau = 1.0 / self.frequency
        elapsed = (s - self.issue) % tau
        return coupon * elapsed / tau

    def to_dict(self) -> Dict[str, Any]:
        return {"face": self.face, "coupon_rate": self.coupon_rate,
                "maturity": self.maturity, "frequency": self.frequency,
                "periods": self.periods}


@dataclass
class InterestRateSwap:
    """固定换浮动利率互换（单曲线口径，浮动腿 = ``N·(df(0)-df(T))``）。"""

    notional: float = 1_000_000.0
    fixed_rate: float = 0.03
    maturity: float = 5.0
    frequency: int = 2
    payer: bool = True                 # True = 支付固定、收取浮动

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError("notional 必须为正")
        if self.maturity <= 0:
            raise ValueError("maturity 必须为正")
        if self.frequency < 1:
            raise ValueError("frequency 必须 >= 1")

    def payment_times(self) -> List[float]:
        n = max(1, int(round(self.maturity * self.frequency)))
        tau = 1.0 / self.frequency
        return [i * tau for i in range(1, n + 1)]

    def fixed_pv(self, curve: DiscountCurve) -> float:
        """固定腿现值（正数）。"""
        tau = 1.0 / self.frequency
        return float(sum(self.notional * self.fixed_rate * tau * float(curve.df(t))
                         for t in self.payment_times()))

    def float_pv(self, curve: DiscountCurve) -> float:
        """浮动腿现值（单曲线、无基差）。"""
        end = self.payment_times()[-1]
        return float(self.notional * (float(curve.df(0.0)) - float(curve.df(end))))

    def pv(self, curve: DiscountCurve) -> float:
        """支付方（payer）为正表示盈利（浮动 - 固定）。"""
        value = self.float_pv(curve) - self.fixed_pv(curve)
        return float(value if self.payer else -value)

    def par_rate(self, curve: DiscountCurve) -> float:
        tau = 1.0 / self.frequency
        annuity = sum(tau * float(curve.df(t)) for t in self.payment_times())
        return (1.0 - float(curve.df(self.payment_times()[-1]))) / annuity

    def dv01(self, curve: DiscountCurve, bp: float = 1.0) -> float:
        """利率上行 ``bp`` 个基点时的价值变化。"""
        return self.pv(curve.shifted(bp)) - self.pv(curve)

    def to_dict(self) -> Dict[str, Any]:
        return {"notional": self.notional, "fixed_rate": self.fixed_rate,
                "maturity": self.maturity, "frequency": self.frequency,
                "payer": self.payer}


# --------------------------------------------------------------------------- #
# 风险：关键利率久期
# --------------------------------------------------------------------------- #
def parallel_dv01(curve: DiscountCurve,
                  price_fn: Callable[[DiscountCurve], float],
                  bp: float = 1.0) -> float:
    """曲线平行上移 ``bp`` 基点时的价值变化。"""
    return float(price_fn(curve.shifted(bp)) - price_fn(curve))


def key_rate_durations(
    curve: DiscountCurve,
    price_fn: Callable[[DiscountCurve], float],
    *,
    bp: float = 1.0,
    buckets: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """逐 pillar 冲击 ``bp`` 基点，返回关键利率久期表。

    ``buckets`` 为空时使用曲线全部 pillar；也可以用 pillar 下标指定子集。
    """
    indices = list(buckets) if buckets is not None else list(range(len(curve.times)))
    base = float(price_fn(curve))
    rows = []
    for i in indices:
        bumped = float(price_fn(curve.key_bumped(i, bp)))
        rows.append({"pillar_index": int(i), "pillar": float(curve.times[i]),
                     "dv01": bumped - base})
    total = float(sum(r["dv01"] for r in rows))
    parallel = parallel_dv01(curve, price_fn, bp)
    return {
        "rows": rows,
        "sum_key_rate_dv01": total,
        "parallel_dv01": parallel,
        "capture_ratio": (total / parallel) if abs(parallel) > 1e-14 else 0.0,
        "unit": f"{bp}bp",
    }


__all__ = [
    "FixedRateBond",
    "InterestRateSwap",
    "key_rate_durations",
    "parallel_dv01",
]
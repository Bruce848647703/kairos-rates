"""数据生成与 CSV 读取。"""
from __future__ import annotations

import csv
from typing import Dict, Iterable, List, Optional, Sequence

from .curve import CurveQuote, DiscountCurve
from .models import NelsonSiegel

DEFAULT_PILLARS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)


def synthetic_curve(
    level: float = 0.030,
    slope: float = -0.015,
    curvature: float = 0.010,
    tau: float = 2.0,
    pillars: Sequence[float] = DEFAULT_PILLARS,
) -> DiscountCurve:
    """用 Nelson-Siegel 生成一条「短低长高」的正常收益率曲线。"""
    model = NelsonSiegel(level, slope, curvature, tau)
    return DiscountCurve.from_zero(list(pillars), model.zero(list(pillars)))


def synthetic_quotes(
    curve: Optional[DiscountCurve] = None,
    tenors: Sequence[float] = (1, 2, 3, 5, 7, 10, 20, 30),
    frequency: int = 2,
) -> List[CurveQuote]:
    """按曲线生成一组「存款 + 平价互换」报价，用于回归 bootstrap。"""
    curve = curve or synthetic_curve()
    # 存款用「简单复利」报价（与 bootstrap 的口径一致），而不是连续复利零息利率
    t_deposit = 0.5
    simple = (1.0 / float(curve.df(t_deposit)) - 1.0) / t_deposit
    quotes = [CurveQuote("deposit", t_deposit, simple)]
    for tenor in tenors:
        quotes.append(CurveQuote("swap", float(tenor),
                                 float(curve.par_rate(tenor, frequency)), frequency))
    return quotes


def load_quotes_csv(path: str) -> List[CurveQuote]:
    """读取曲线报价 CSV：``kind, maturity, rate, frequency``（列名大小写不敏感）。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} 没有数据")
    quotes: List[CurveQuote] = []
    for row in rows:
        lowered = {str(k).strip().lower(): v for k, v in row.items()}
        quotes.append(CurveQuote(
            kind=str(lowered.get("kind", "swap")),
            maturity=float(lowered.get("maturity", lowered.get("tenor", "0"))),
            rate=float(lowered.get("rate", "0")),
            frequency=int(float(lowered.get("frequency", 2)))))
    return quotes


def save_quotes_csv(quotes: Iterable[CurveQuote], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "maturity", "rate", "frequency"])
        for q in quotes:
            writer.writerow([q.kind, q.maturity, q.rate, q.frequency])


def curve_table(curve: DiscountCurve,
                tenors: Sequence[float] = DEFAULT_PILLARS) -> List[Dict[str, float]]:
    """给 CLI/表格用的曲线摘要。"""
    rows = []
    for t in tenors:
        rows.append({
            "tenor": float(t),
            "zero": float(curve.zero(t)),
            "df": float(curve.df(t)),
            "fwd_3m": float(curve.forward(t, t + 0.25)) if t + 0.25 <= curve.times[-1] else float("nan"),
        })
    return rows


__all__ = [
    "DEFAULT_PILLARS",
    "curve_table",
    "load_quotes_csv",
    "save_quotes_csv",
    "synthetic_curve",
    "synthetic_quotes",
]
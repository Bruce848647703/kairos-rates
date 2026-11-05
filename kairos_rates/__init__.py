"""Kairos Rates —— 自研利率研究库。

覆盖固定收益的核心链路：

- **曲线**：贴现曲线（对数线性插值）、存款/互换逐点 bootstrap、平价利率、远期利率
- **拟合**：Nelson-Siegel / Svensson（网格 + 线性最小二乘 + Nelder-Mead）
- **短期利率模型**：Vasicek、CIR（解析债券价格）、Hull-White 1F（精确重定价初始曲线）
- **工具定价**：固定利率债券（价格/YTM/久期/凸性/应计）、利率互换（PV/平价/DV01）
- **风险**：平行 DV01、关键利率久期（逐 pillar 冲击）

快速开始::

    pip install -e .
    python -m kairos_rates            # 完整演示
    python -m kairos_rates bond --coupon 0.03 --maturity 10
"""
from __future__ import annotations

from .curve import CurveQuote, DiscountCurve, bootstrap
from .data import (
    curve_table,
    load_quotes_csv,
    save_quotes_csv,
    synthetic_curve,
    synthetic_quotes,
)
from .daycount import add_months, add_years, schedule, to_date, year_fraction
from .instruments import (
    FixedRateBond,
    InterestRateSwap,
    key_rate_durations,
    parallel_dv01,
)
from .models import (
    CIR,
    HullWhite,
    NelsonSiegel,
    Svensson,
    Vasicek,
    fit_nelson_siegel,
    fit_svensson,
)

__version__ = "0.1.1"

__all__ = [
    "CIR",
    "CurveQuote",
    "DiscountCurve",
    "FixedRateBond",
    "HullWhite",
    "InterestRateSwap",
    "NelsonSiegel",
    "Svensson",
    "Vasicek",
    "__version__",
    "add_months",
    "add_years",
    "bootstrap",
    "curve_table",
    "fit_nelson_siegel",
    "fit_svensson",
    "key_rate_durations",
    "load_quotes_csv",
    "parallel_dv01",
    "save_quotes_csv",
    "schedule",
    "synthetic_curve",
    "synthetic_quotes",
    "to_date",
    "year_fraction",
]
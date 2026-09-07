"""示例 3：债券与互换定价 + 关键利率久期。

运行：python examples/03_bond_swap_risk.py
"""
from __future__ import annotations

from kairos_rates import (
    FixedRateBond,
    InterestRateSwap,
    bootstrap,
    key_rate_durations,
    synthetic_curve,
    synthetic_quotes,
)


def main() -> None:
    curve = bootstrap(synthetic_quotes(synthetic_curve()))

    bond = FixedRateBond(100.0, 0.03, 10.0, 2)
    price = bond.price(curve)
    print(f"10Y 3% 债券  价格 {price:.4f}  YTM {bond.ytm(price):.4%}  "
          f"久期 {bond.macaulay_duration(curve):.3f}  凸性 {bond.convexity(curve):.1f}")

    par = curve.par_rate(5.0, 2)
    swap = InterestRateSwap(1_000_000.0, par, 5.0, 2, payer=True)
    print(f"5Y 平价互换  平价利率 {par:.4%}  PV {swap.pv(curve):.2f}  DV01 {swap.dv01(curve):.2f}")

    print("\n关键利率久期（10Y 债券，1bp）：")
    report = key_rate_durations(curve, bond.price, bp=1.0)
    for row in report["rows"]:
        if abs(row["dv01"]) > 1e-6:
            print(f"  {row['pillar']:>5.1f}y  {row['dv01']:+.5f}")
    print(f"  合计 {report['sum_key_rate_dv01']:+.5f}  "
          f"平行 {report['parallel_dv01']:+.5f}  捕获率 {report['capture_ratio']:.1%}")


if __name__ == "__main__":
    main()

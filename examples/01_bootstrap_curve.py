"""示例 1：从「存款 + 平价互换」报价 bootstrap 出一条贴现曲线。

运行：python examples/01_bootstrap_curve.py
"""
from __future__ import annotations

from kairos_rates import bootstrap, curve_table, synthetic_curve, synthetic_quotes


def main() -> None:
    truth = synthetic_curve()
    quotes = synthetic_quotes(truth)
    curve = bootstrap(quotes)

    print(f"{'期限':>7}{'零息利率':>11}{'贴现因子':>12}")
    for row in curve_table(curve):
        print(f"{row['tenor']:>7.1f}{row['zero']:>11.2%}{row['df']:>12.6f}")

    err = max(abs(float(curve.df(q.maturity)) - 1.0 / (1.0 + q.rate * q.maturity))
              for q in quotes if q.kind == "deposit")
    swap_err = max(abs(curve.par_rate(q.maturity, q.frequency) - q.rate)
                   for q in quotes if q.kind == "swap")
    print(f"\n报价重定价误差：存款 {err:.1e}，互换 {swap_err:.1e}")
    print(f"相对平滑真曲线的最大零息误差："
          f"{max(abs(float(curve.zero(t)) - float(truth.zero(t))) for t in curve.pillars):.2e}")


if __name__ == "__main__":
    main()

"""命令行入口：``kairos-rates`` / ``python -m kairos_rates``。"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np

from .curve import DiscountCurve, bootstrap
from .data import curve_table, load_quotes_csv, synthetic_curve, synthetic_quotes
from .instruments import FixedRateBond, InterestRateSwap, key_rate_durations
from .models import CIR, HullWhite, Vasicek, fit_nelson_siegel, fit_svensson


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kairos-rates",
        description="利率研究：曲线 bootstrap、NS/Svensson、Vasicek/CIR/Hull-White、债券/互换与关键利率久期。")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="完整演示（默认）")

    curve = sub.add_parser("curve", help="从 CSV bootstrap 曲线")
    curve.add_argument("--csv", required=True)
    curve.add_argument("--steps", type=int, default=8)

    bond = sub.add_parser("bond", help="债券定价（用合成曲线）")
    bond.add_argument("--face", type=float, default=100.0)
    bond.add_argument("--coupon", type=float, default=0.03)
    bond.add_argument("--maturity", type=float, default=10.0)
    bond.add_argument("--frequency", type=int, default=2)

    swap = sub.add_parser("swap", help="利率互换定价（用合成曲线）")
    swap.add_argument("--notional", type=float, default=1_000_000.0)
    swap.add_argument("--rate", type=float, default=None, help="固定利率，缺省为平价利率")
    swap.add_argument("--maturity", type=float, default=5.0)
    swap.add_argument("--frequency", type=int, default=2)
    swap.add_argument("--receive", action="store_true", help="收取固定（默认支付固定）")

    return parser


def _print_curve(curve: DiscountCurve, tenors=None) -> None:
    print(f"{'期限':>7}{'零息利率':>11}{'贴现因子':>12}{'3M远期':>10}")
    print("-" * 42)
    for row in curve_table(curve, tenors or curve.pillars):
        fwd = row["fwd_3m"]
        fwd_text = f"{fwd:>10.2%}" if np.isfinite(fwd) else f"{'—':>10}"
        print(f"{row['tenor']:>7.1f}{row['zero']:>11.2%}{row['df']:>12.6f}{fwd_text}")


def _cmd_demo(_: argparse.Namespace) -> int:
    original = synthetic_curve()
    quotes = synthetic_quotes(original)
    boot = bootstrap(quotes)

    print("=" * 66)
    print("1) 合成报价 -> bootstrap 曲线（与真曲线对比）")
    print("=" * 66)
    _print_curve(boot)
    max_err = max(abs(float(boot.zero(t)) - float(original.zero(t))) for t in boot.pillars)
    print(f"\n最大零息误差（相对原曲线）：{max_err:.2e}")

    print()
    print("=" * 66)
    print("2) 参数化拟合")
    print("=" * 66)
    ts = np.array(boot.pillars)
    zs = np.asarray(boot.zero(ts))
    ns, rmse_ns = fit_nelson_siegel(ts, zs)
    sv, rmse_sv = fit_svensson(ts, zs)
    print(f"  Nelson-Siegel: beta0={ns.beta0:.4f} beta1={ns.beta1:+.4f} "
          f"beta2={ns.beta2:+.4f} tau={ns.tau:.3f}  RMSE={rmse_ns*1e4:.2f}bp")
    print(f"  Svensson:      beta0={sv.beta0:.4f} beta1={sv.beta1:+.4f} "
          f"beta2={sv.beta2:+.4f} beta3={sv.beta3:+.4f} tau1={sv.tau1:.2f} tau2={sv.tau2:.2f}  "
          f"RMSE={rmse_sv*1e4:.2f}bp")

    print()
    print("=" * 66)
    print("3) 债券：定价、YTM、久期、凸性")
    print("=" * 66)
    bond = FixedRateBond(face=100.0, coupon_rate=0.03, maturity=10.0, frequency=2)
    price = bond.price(boot)
    print(f"  10Y 3% 债券  价格={price:.4f}  YTM={bond.ytm(price):.4%}  "
          f"麦考利久期={bond.macaulay_duration(boot):.3f}  "
          f"凸性={bond.convexity(boot):.3f}")

    print()
    print("=" * 66)
    print("4) 互换：平价利率与 DV01")
    print("=" * 66)
    par = boot.par_rate(5.0, 2)
    swap = InterestRateSwap(notional=1_000_000.0, fixed_rate=par, maturity=5.0,
                            frequency=2, payer=True)
    print(f"  5Y 平价利率={par:.4%}  平价互换 PV={swap.pv(boot):.2f}  "
          f"DV01={swap.dv01(boot):.2f}")

    print()
    print("=" * 66)
    print("5) 关键利率久期（10Y 债券，1bp 冲击）")
    print("=" * 66)
    report = key_rate_durations(boot, bond.price, bp=1.0)
    print(f"  {'pillar':>10}{'DV01':>12}")
    for row in report["rows"]:
        print(f"  {row['pillar']:>10.1f}{row['dv01']:>12.4f}")
    print(f"  合计 {report['sum_key_rate_dv01']:.4f}  "
          f"平行 {report['parallel_dv01']:.4f}  捕获率 {report['capture_ratio']:.1%}")

    print()
    print("=" * 66)
    print("6) 短期利率模型（1Y 零息利率对比）")
    print("=" * 66)
    vasicek = Vasicek(kappa=0.8, theta=0.035, sigma=0.01, r0=float(boot.zero(0.5)))
    cir = CIR(kappa=0.8, theta=0.035, sigma=0.08, r0=float(boot.zero(0.5)))
    hw = HullWhite(a=0.1, sigma=0.01, curve=boot)
    print(f"  Vasicek 1Y={vasicek.zero_rate(1.0):.4%}  "
          f"CIR 1Y={cir.zero_rate(1.0):.4%}  "
          f"Hull-White 1Y={hw.zero_rate(1.0):.4%}（与市场 {float(boot.zero(1.0)):.4%} 一致）")
    return 0


def _cmd_curve(args: argparse.Namespace) -> int:
    quotes = load_quotes_csv(args.csv)
    curve = bootstrap(quotes)
    print(f"读取 {len(quotes)} 条报价")
    _print_curve(curve)
    return 0


def _cmd_bond(args: argparse.Namespace) -> int:
    curve = bootstrap(synthetic_quotes())
    bond = FixedRateBond(args.face, args.coupon, args.maturity, args.frequency)
    price = bond.price(curve)
    print(f"面值 {args.face:.2f}  票息 {args.coupon:.2%}  期限 {args.maturity:g}y  "
          f"频率 {args.frequency}")
    print(f"  价格       {price:.6f}")
    print(f"  YTM        {bond.ytm(price):.4%}")
    print(f"  麦考利久期 {bond.macaulay_duration(curve):.4f}")
    print(f"  凸性       {bond.convexity(curve):.4f}")
    print(f"  应计利息   {bond.accrued_interest(args.maturity * 0.5):.4f}")
    return 0


def _cmd_swap(args: argparse.Namespace) -> int:
    curve = bootstrap(synthetic_quotes())
    par = curve.par_rate(args.maturity, args.frequency)
    rate = par if args.rate is None else args.rate
    swap = InterestRateSwap(args.notional, rate, args.maturity, args.frequency,
                            payer=not args.receive)
    print(f"名义 {args.notional:,.0f}  期限 {args.maturity:g}y  频率 {args.frequency}  "
          f"{'收固定' if args.receive else '付固定'}")
    print(f"  平价利率 {par:.4%}   固定利率 {rate:.4%}")
    print(f"  PV       {swap.pv(curve):,.2f}")
    print(f"  DV01     {swap.dv01(curve):,.2f}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not argv_list or argv_list[0].startswith("-"):
        args = parser.parse_args(["demo"] + argv_list)
    else:
        args = parser.parse_args(argv_list)
    command = args.command or "demo"
    return {
        "demo": _cmd_demo,
        "curve": _cmd_curve,
        "bond": _cmd_bond,
        "swap": _cmd_swap,
    }[command](args)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
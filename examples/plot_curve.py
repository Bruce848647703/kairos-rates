"""生成 README 用图（需要 matplotlib：pip install -e ".[plot]"）。"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kairos_rates import (  # noqa: E402
    FixedRateBond,
    bootstrap,
    fit_nelson_siegel,
    key_rate_durations,
    synthetic_curve,
    synthetic_quotes,
)


def main(out: str = "docs/yield-curve.png") -> None:
    truth = synthetic_curve()
    curve = bootstrap(synthetic_quotes(truth))
    ts = np.linspace(curve.times[0], curve.times[-1], 200)
    ns, _ = fit_nelson_siegel(np.array(curve.pillars), np.asarray(curve.zero(curve.pillars)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax1.plot(ts, np.asarray(curve.zero(ts)) * 100, color="#1f4e79", lw=2.2,
             label="bootstrapped zero curve")
    ax1.plot(ts, np.asarray(ns.zero(ts)) * 100, "--", color="#c0392b", lw=1.4,
             label="Nelson-Siegel fit")
    ax1.scatter(curve.pillars, np.asarray(curve.zero(curve.pillars)) * 100,
                color="#1f4e79", zorder=5, s=22)
    ax1.set_xlabel("maturity (years)")
    ax1.set_ylabel("zero rate (%)")
    ax1.set_title("Yield curve")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    bond = FixedRateBond(100.0, 0.03, 10.0, 2)
    report = key_rate_durations(curve, bond.price, bp=1.0)
    labels = [f"{r['pillar']:g}" for r in report["rows"]]
    values = [r["dv01"] for r in report["rows"]]
    ax2.bar(labels, values, color="#2e86c1")
    ax2.set_xlabel("curve pillar (years)")
    ax2.set_ylabel("key-rate DV01 (price points per 1bp)")
    ax2.set_title("10Y bond key-rate risk")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()

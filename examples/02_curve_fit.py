"""示例 2：给一条带扰动的曲线拟合 Nelson-Siegel 与 Svensson。

运行：python examples/02_curve_fit.py
"""
from __future__ import annotations

import numpy as np

from kairos_rates import fit_nelson_siegel, fit_svensson, synthetic_curve

PILLARS = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])


def main() -> None:
    truth = synthetic_curve()
    rng = np.random.default_rng(7)
    observed = np.asarray(truth.zero(PILLARS)) + rng.normal(0.0, 0.0008, PILLARS.size)

    ns, rmse_ns = fit_nelson_siegel(PILLARS, observed)
    sv, rmse_sv = fit_svensson(PILLARS, observed)

    print(f"{'期限':>7}{'观测':>10}{'NS':>10}{'Svensson':>11}")
    for t, y in zip(PILLARS, observed):
        print(f"{t:>7.1f}{y:>10.3%}{float(ns.zero(t)):>10.3%}{float(sv.zero(t)):>11.3%}")
    print(f"\nNS RMSE       = {rmse_ns*1e4:.2f} bp   params={ns.to_dict()}")
    print(f"Svensson RMSE = {rmse_sv*1e4:.2f} bp   params={sv.to_dict()}")


if __name__ == "__main__":
    main()

"""数值优化小工具：Nelder-Mead 单纯形（无第三方依赖、确定性）。

曲面校准的目标函数（加权 SSE）没有解析梯度，且参数维度低（3~5），
Nelder-Mead 是最省心且完全可复现的选择。
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def nelder_mead(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    step: float = 0.1,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """最小化 ``objective``，返回 ``(最优参数, 最优值)``。"""
    x0 = np.asarray(x0, dtype="float64").ravel()
    n = x0.size
    if n == 0:
        raise ValueError("x0 不能为空")

    simplex = [x0.copy()]
    for i in range(n):
        point = x0.copy()
        point[i] += step if point[i] == 0 else step * (1.0 + abs(point[i]))
        simplex.append(point)
    values = [float(objective(p)) for p in simplex]

    for _ in range(int(max_iter)):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) <= tol * (1.0 + abs(values[0])):
            break

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        reflected = centroid + (centroid - worst)
        f_reflected = float(objective(reflected))
        if f_reflected < values[0]:
            expanded = centroid + 2.0 * (centroid - worst)
            f_expanded = float(objective(expanded))
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
        else:
            contracted = centroid + 0.5 * (worst - centroid)
            f_contracted = float(objective(contracted))
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = float(objective(simplex[i]))

    best = int(np.argmin(values))
    return simplex[best].copy(), float(values[best])


def weighted_sse(predicted: np.ndarray, observed: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
    """加权残差平方和（忽略 NaN）。"""
    diff = np.asarray(predicted, dtype="float64") - np.asarray(observed, dtype="float64")
    mask = np.isfinite(diff)
    if weights is None:
        return float(np.sum(diff[mask] ** 2))
    w = np.asarray(weights, dtype="float64")[mask]
    return float(np.sum(w * diff[mask] ** 2))


__all__ = ["nelder_mead", "weighted_sse"]
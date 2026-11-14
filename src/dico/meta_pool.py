"""Reliability-weighted, reversal-equivariant multi-view pooling.

The meta-decoder is deliberately small: one calibrated log-probability tensor
per view family, non-negative weights constrained to the probability simplex,
and a centered class bias.  With fixed base predictions its regularized NLL is
convex in the direct weight/bias parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass(frozen=True)
class MetaPoolModel:
    view_names: tuple[str, ...]
    temperatures: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    l2: float

    def to_dict(self) -> dict:
        return {
            "view_names": list(self.view_names),
            "temperatures": dict(zip(self.view_names, map(float, self.temperatures))),
            "weights": dict(zip(self.view_names, map(float, self.weights))),
            "bias": list(map(float, self.bias)),
            "l2": float(self.l2),
        }


def fit_temperatures(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit one positive scalar temperature for each [N, view, class] slice."""
    logp = np.asarray(logp, dtype=float)
    y = np.asarray(y, dtype=int)
    temperatures = np.ones(logp.shape[1])
    for view in range(logp.shape[1]):
        scores = logp[:, view]

        def nll(t: float) -> float:
            p = softmax(scores / t)
            return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())

        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"temperature fit failed for view {view}: {result.message}")
        temperatures[view] = result.x
    return temperatures


def calibrated_scores(logp: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    logp = np.asarray(logp, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    if logp.ndim != 3 or temperatures.shape != (logp.shape[1],):
        raise ValueError("expected logp [N, views, classes] and one temperature per view")
    return logp / temperatures[None, :, None]


def fit_meta_pool(
    logp: np.ndarray,
    y: np.ndarray,
    view_names: list[str] | tuple[str, ...],
    *,
    l2: float = 0.1,
    calibrate: bool = True,
) -> MetaPoolModel:
    """Fit convex simplex weights and centered class biases by NLL."""
    logp = np.asarray(logp, dtype=float)
    y = np.asarray(y, dtype=int)
    if logp.ndim != 3 or len(y) != len(logp):
        raise ValueError("logp must be [N, views, classes] and align with y")
    n_views, n_classes = logp.shape[1:]
    if len(view_names) != n_views:
        raise ValueError("view_names must match the view dimension")
    temperatures = fit_temperatures(logp, y) if calibrate else np.ones(n_views)
    x = calibrated_scores(logp, temperatures)
    uniform = np.full(n_views, 1.0 / n_views)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = params[:n_views]
        raw_bias = params[n_views:]
        bias = raw_bias - raw_bias.mean()
        return weights, bias

    def objective(params: np.ndarray) -> float:
        weights, bias = unpack(params)
        scores = np.einsum("nvc,v->nc", x, weights) + bias
        p = softmax(scores)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()
        reg = l2 * ((weights - uniform) ** 2).sum() + l2 * (bias**2).sum()
        return float(nll + reg)

    initial = np.concatenate([uniform, np.zeros(n_classes)])
    bounds = [(0.0, 1.0)] * n_views + [(None, None)] * n_classes
    constraints = {"type": "eq", "fun": lambda p: float(p[:n_views].sum() - 1.0)}
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"convex meta-pool fit failed: {result.message}")
    weights, bias = unpack(result.x)
    return MetaPoolModel(tuple(view_names), temperatures, weights, bias, l2)


def predict_meta_pool(logp: np.ndarray, model: MetaPoolModel) -> np.ndarray:
    x = calibrated_scores(logp, model.temperatures)
    return (np.einsum("nvc,v->nc", x, model.weights) + model.bias).argmax(axis=1)


def uniform_pool(logp: np.ndarray) -> np.ndarray:
    return np.asarray(logp).mean(axis=1).argmax(axis=1)

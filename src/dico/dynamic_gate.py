"""Uncertainty-conditioned, reversal-compatible routing over frozen experts.

The caller first maps both directions of every semantic pair to one canonical
orientation.  This module then learns sample-wise simplex weights from the
experts' calibrated score geometry.  Expanding the canonical prediction with
``REV_PERM`` makes the complete decoder exactly reversal equivariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .meta_pool import calibrated_scores, fit_meta_pool, softmax


def gate_features(scores: np.ndarray) -> np.ndarray:
    """Build label-aware confidence/agreement features from [N,V,C] scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 3:
        raise ValueError("scores must have shape [examples, views, classes]")
    probs = softmax(scores, axis=2)
    centered = scores - scores.mean(axis=2, keepdims=True)
    ordered = np.sort(probs, axis=2)
    confidence = ordered[:, :, -1]
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=2)
    pred = probs.argmax(axis=2)
    votes = np.stack([(pred == c).mean(axis=1) for c in range(scores.shape[2])], axis=1)
    agreement = np.take_along_axis(votes[:, None, :], pred[:, :, None], axis=2).squeeze(2)
    return np.concatenate(
        [
            centered.reshape(len(scores), -1),
            probs.reshape(len(scores), -1),
            confidence,
            margin,
            entropy,
            votes,
            agreement,
        ],
        axis=1,
    )


@dataclass(frozen=True)
class DynamicGateModel:
    view_names: tuple[str, ...]
    temperatures: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    gate_coef: np.ndarray
    gate_bias: np.ndarray
    class_bias: np.ndarray
    prior_weights: np.ndarray
    l2: float
    anchor: float

    def to_dict(self) -> dict:
        return {
            "view_names": list(self.view_names),
            "temperatures": list(map(float, self.temperatures)),
            "feature_mean": list(map(float, self.feature_mean)),
            "feature_scale": list(map(float, self.feature_scale)),
            "gate_coef": self.gate_coef.tolist(),
            "gate_bias": list(map(float, self.gate_bias)),
            "class_bias": list(map(float, self.class_bias)),
            "prior_weights": dict(zip(self.view_names, map(float, self.prior_weights))),
            "l2": float(self.l2),
            "anchor": float(self.anchor),
        }


def _unpack(params: np.ndarray, n_features: int, n_views: int, n_classes: int):
    split = n_features * n_views
    coef = params[:split].reshape(n_features, n_views)
    gate_bias = params[split : split + n_views]
    raw_class_bias = params[split + n_views : split + n_views + n_classes]
    class_bias = raw_class_bias - raw_class_bias.mean()
    return coef, gate_bias, class_bias


def fit_dynamic_gate(
    logp: np.ndarray,
    y: np.ndarray,
    view_names: list[str] | tuple[str, ...],
    *,
    l2: float = 1e-2,
    anchor: float = 1e-1,
    maxiter: int = 700,
) -> DynamicGateModel:
    """Fit a linear soft router by differentiable mixture NLL.

    ``anchor`` shrinks per-example routing weights toward the calibrated static
    pool.  This is important when the number of semantic pairs is small.
    """
    logp = np.asarray(logp, dtype=float)
    y = np.asarray(y, dtype=int)
    if logp.ndim != 3 or len(logp) != len(y):
        raise ValueError("logp must be [N,V,C] and align with y")
    n, n_views, n_classes = logp.shape
    if len(view_names) != n_views:
        raise ValueError("view_names must match the view dimension")

    prior = fit_meta_pool(logp, y, view_names, l2=0.1, calibrate=True)
    scores = calibrated_scores(logp, prior.temperatures)
    raw_features = gate_features(scores)
    feature_mean = raw_features.mean(axis=0)
    feature_scale = raw_features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    features = (raw_features - feature_mean) / feature_scale
    n_features = features.shape[1]
    target = np.eye(n_classes)[y]

    initial = np.concatenate(
        [
            np.zeros(n_features * n_views),
            np.log(np.clip(prior.weights, 1e-8, 1.0)),
            prior.bias,
        ]
    )

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        coef, gate_bias, class_bias = _unpack(params, n_features, n_views, n_classes)
        route_logits = features @ coef + gate_bias
        weights = softmax(route_logits, axis=1)
        mixed_scores = np.einsum("nv,nvc->nc", weights, scores) + class_bias
        probs = softmax(mixed_scores, axis=1)
        nll = -np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean()
        shrink = anchor * ((weights - prior.weights[None, :]) ** 2).mean()
        reg = l2 * (np.mean(coef**2) + np.mean(class_bias**2))

        grad_scores = (probs - target) / n
        grad_weights = np.einsum("nc,nvc->nv", grad_scores, scores)
        grad_weights += 2.0 * anchor * (weights - prior.weights[None, :]) / weights.size
        grad_route = weights * (grad_weights - (grad_weights * weights).sum(axis=1, keepdims=True))
        grad_coef = features.T @ grad_route + 2.0 * l2 * coef / coef.size
        grad_gate_bias = grad_route.sum(axis=0)
        grad_class_bias = grad_scores.sum(axis=0) + 2.0 * l2 * class_bias / n_classes
        grad_class_bias -= grad_class_bias.mean()
        grad = np.concatenate([grad_coef.ravel(), grad_gate_bias, grad_class_bias])
        return float(nll + shrink + reg), grad

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success and result.status != 1:
        raise RuntimeError(f"dynamic gate fit failed: {result.message}")
    coef, gate_bias, class_bias = _unpack(result.x, n_features, n_views, n_classes)
    return DynamicGateModel(
        tuple(view_names),
        prior.temperatures,
        feature_mean,
        feature_scale,
        coef,
        gate_bias,
        class_bias,
        prior.weights,
        l2,
        anchor,
    )


def routing_weights(logp: np.ndarray, model: DynamicGateModel) -> np.ndarray:
    scores = calibrated_scores(logp, model.temperatures)
    features = (gate_features(scores) - model.feature_mean) / model.feature_scale
    return softmax(features @ model.gate_coef + model.gate_bias, axis=1)


def predict_dynamic_gate(logp: np.ndarray, model: DynamicGateModel) -> np.ndarray:
    scores = calibrated_scores(logp, model.temperatures)
    weights = routing_weights(logp, model)
    return (np.einsum("nv,nvc->nc", weights, scores) + model.class_bias).argmax(axis=1)

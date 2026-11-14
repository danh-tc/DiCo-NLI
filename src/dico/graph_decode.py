"""Post-hoc constrained MAP inference over observed phrase-relation graphs.

Nodes are exact ``(language, phrase)`` pairs.  No latent edges are created: a
transitivity rule is instantiated only when all three directed edges are rows
in the input dataframe.  This deliberately avoids conflating translations or
unobserved senses and keeps the MILPs small.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import BWD, EQ, FWD, LABEL2ID, LABELS, NEG, REV_PERM


# Only deterministic entries of the collapsed four-relation composition table.
COMPOSITION = {
    (EQ, EQ): EQ,
    (EQ, FWD): FWD,
    (EQ, BWD): BWD,
    (EQ, NEG): NEG,
    (FWD, EQ): FWD,
    (FWD, FWD): FWD,
    (BWD, EQ): BWD,
    (BWD, BWD): BWD,
    (NEG, EQ): NEG,
}


@dataclass(frozen=True)
class GraphStats:
    components: int = 0
    solved_components: int = 0
    observed_paths: int = 0
    reversal_pairs: int = 0


def temperature_scale(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply scalar temperature scaling when only probabilities were saved.

    ``log(probs)`` is a valid set of logits up to a row-wise additive constant,
    so this is equivalent to softmax(original_logits / temperature).
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    logp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    scaled = np.exp(logp)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray | list[int],
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit one temperature by minimizing NLL on a calibration split."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(labels, dtype=int)
    if len(y) != len(probs):
        raise ValueError("labels and probs must have the same number of rows")

    def nll(t: float) -> float:
        p = temperature_scale(probs, float(t))
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature fitting failed: {result.message}")
    return float(result.x)


def _node(lang, text) -> tuple[str, str]:
    return str(lang), str(text)


def _component_rows(df: pd.DataFrame) -> list[list[int]]:
    """Return weakly connected components as dataframe row positions."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    endpoints = []
    for row in df.itertuples(index=False):
        a = _node(row.text1_lang, row.text1)
        b = _node(row.text2_lang, row.text2)
        union(a, b)
        endpoints.append((a, b))

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (a, _) in enumerate(endpoints):
        groups[find(a)].append(i)
    return list(groups.values())


def _observed_paths(df: pd.DataFrame, rows: list[int]) -> list[tuple[int, int, int]]:
    """Find row triples for observed A->B, B->C, A->C paths."""
    edge_rows: dict[tuple[tuple[str, str], tuple[str, str]], list[int]] = defaultdict(list)
    outgoing: dict[tuple[str, str], list[tuple[tuple[str, str], int]]] = defaultdict(list)
    endpoints = {}
    for i in rows:
        row = df.iloc[i]
        a = _node(row.text1_lang, row.text1)
        b = _node(row.text2_lang, row.text2)
        endpoints[i] = (a, b)
        edge_rows[(a, b)].append(i)
        outgoing[a].append((b, i))

    paths = []
    for i in rows:
        a, b = endpoints[i]
        for c, j in outgoing.get(b, []):
            if a == b or b == c or a == c:
                continue
            for k in edge_rows.get((a, c), []):
                paths.append((i, j, k))
    return paths


def _solve_component(
    df: pd.DataFrame,
    logp: np.ndarray,
    rows: list[int],
    transitivity_penalty: float | None,
    time_limit: float | None,
) -> tuple[np.ndarray, int, int]:
    """Solve one component; ``None`` penalty means hard transitivity."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    local = {row: i for i, row in enumerate(rows)}
    paths = _observed_paths(df, rows)
    reverse_pairs = []
    if "rev_idx" in df:
        for row in rows:
            rev = int(df.iloc[row].rev_idx)
            if rev in local and row < rev:
                reverse_pairs.append((row, rev))

    # No coupled constraints: exact row-wise MAP needs no solver.
    if not paths and not reverse_pairs:
        return logp[rows].argmax(axis=1), 0, 0

    n_edges = len(rows)
    n_z = n_edges * len(LABELS)
    soft = transitivity_penalty is not None
    n_vars = n_z + (len(paths) if soft else 0)
    objective = np.zeros(n_vars)
    objective[:n_z] = -logp[rows].reshape(-1)  # scipy minimizes negative log-likelihood
    if soft:
        if transitivity_penalty < 0 or not np.isfinite(transitivity_penalty):
            raise ValueError("transitivity_penalty must be finite and non-negative")
        objective[n_z:] = transitivity_penalty

    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coeffs: list[tuple[int, float]], lb: float, ub: float) -> None:
        r = len(lower)
        for col, value in coeffs:
            rr.append(r)
            cc.append(col)
            vv.append(value)
        lower.append(lb)
        upper.append(ub)

    def z(row: int, label: int) -> int:
        return local[row] * len(LABELS) + label

    for row in rows:
        add([(z(row, c), 1.0) for c in range(len(LABELS))], 1.0, 1.0)

    for row, rev in reverse_pairs:
        for c, rc in enumerate(REV_PERM):
            add([(z(row, c), 1.0), (z(rev, rc), -1.0)], 0.0, 0.0)

    rules = [(LABEL2ID[a], LABEL2ID[b], LABEL2ID[c]) for (a, b), c in COMPOSITION.items()]
    for path_idx, (e1, e2, e3) in enumerate(paths):
        slack = n_z + path_idx
        for c1, c2, c3 in rules:
            coeffs = [(z(e1, c1), 1.0), (z(e2, c2), 1.0), (z(e3, c3), -1.0)]
            if soft:
                coeffs.append((slack, -1.0))
            add(coeffs, -np.inf, 1.0)

    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), n_vars)).tocsr()
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[:n_z] = 1
    options = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"graph MILP failed (status={result.status}): {result.message}")
    labels = result.x[:n_z].reshape(n_edges, len(LABELS)).argmax(axis=1)
    return labels, len(paths), len(reverse_pairs)


def graph_map(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    transitivity_penalty: float | None = 1.0,
    temperature: float = 1.0,
    time_limit: float | None = None,
    return_stats: bool = False,
) -> list[str] | tuple[list[str], GraphStats]:
    """Decode with hard reversal and soft/hard observed transitivity.

    ``transitivity_penalty=None`` makes transitivity hard.  Otherwise each
    violated observed A->B->C rule costs that non-negative amount.
    """
    if len(df) != len(probs) or np.asarray(probs).shape != (len(df), len(LABELS)):
        raise ValueError(f"expected probs shape ({len(df)}, {len(LABELS)})")
    scaled = temperature_scale(probs, temperature)
    logp = np.log(np.clip(scaled, 1e-12, 1.0))
    out = np.empty(len(df), dtype=int)
    components = _component_rows(df)
    solved = paths_total = pairs_total = 0
    for rows in components:
        labels, paths, pairs = _solve_component(
            df, logp, rows, transitivity_penalty, time_limit
        )
        out[rows] = labels
        paths_total += paths
        pairs_total += pairs
        solved += int(paths > 0 or pairs > 0)
    result = [LABELS[i] for i in out]
    stats = GraphStats(len(components), solved, paths_total, pairs_total)
    return (result, stats) if return_stats else result


def graph_soft(df: pd.DataFrame, probs: np.ndarray) -> list[str]:
    return graph_map(df, probs, transitivity_penalty=1.0)


def graph_hard(df: pd.DataFrame, probs: np.ndarray) -> list[str]:
    return graph_map(df, probs, transitivity_penalty=None)

"""Turn class probabilities into labels.

All decoders take `df` (with `rev_idx`, see data.add_reverse_index) and `probs`
of shape (n, 4) aligned with df rows, and return a list of label strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABEL2ID, LABELS, NEG, REV_PERM
from .graph_decode import graph_hard, graph_map, graph_soft


def _logp(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, 1e-12, 1.0))


def argmax(df: pd.DataFrame, probs: np.ndarray) -> list[str]:
    return [LABELS[i] for i in probs.argmax(axis=1)]


def _joint(df: pd.DataFrame, logp: np.ndarray, unpaired) -> list[str]:
    """Decode each reverse pair ONCE: l* = argmax log p(l|A,B) + log p(Rev(l)|B,A),
    then assign l* to (A,B) and Rev(l*) to (B,A). Deciding per pair (not per row)
    keeps the two directions consistent even when scores tie."""
    rev = df["rev_idx"].to_numpy()
    out: list[str | None] = [None] * len(df)
    for i, j in enumerate(rev):
        if out[i] is not None:
            continue
        if j < 0:
            out[i] = unpaired(logp[i])
        else:
            label = LABELS[int((logp[i] + logp[j][REV_PERM]).argmax())]
            out[i], out[j] = label, LABELS[REV_PERM[LABEL2ID[label]]]
    return out


def joint(df: pd.DataFrame, probs: np.ndarray) -> list[str]:
    """Consistent pair decoding. Guarantees SoftCons = 1.0 (rows without a partner: argmax)."""
    return _joint(df, _logp(probs), lambda lp: LABELS[int(lp.argmax())])


def joint_structural(df: pd.DataFrame, probs: np.ndarray) -> list[str]:
    """EXPERIMENTAL, uses a data artifact: in train/dev, NEGATIVE_OTHER rows (and only
    they) have no reverse partner. Rows without a partner -> NEG; paired rows -> joint
    over the three reversible labels.

    Ask the organizers before using this in a submission; report it as an artifact.
    """
    logp = _logp(probs)
    logp[:, LABEL2ID[NEG]] = -np.inf
    return _joint(df, logp, lambda lp: NEG)


def swap_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows with (text1, text2) and their languages swapped: the input for p(.|B,A)."""
    out = df.copy()
    out["text1"], out["text2"] = df["text2"].to_numpy(), df["text1"].to_numpy()
    if {"text1_lang", "text2_lang"} <= set(df.columns):
        out["text1_lang"], out["text2_lang"] = df["text2_lang"].to_numpy(), df["text1_lang"].to_numpy()
    return out


def self_swap_scores(probs: np.ndarray, probs_swap: np.ndarray, neg_bias: float = 0.0) -> np.ndarray:
    """score(l) = log p(l|A,B) + log p(Rev(l)|B,A) + neg_bias * [l = NEG]."""
    if probs.shape != probs_swap.shape:
        raise ValueError(f"probs {probs.shape} and probs_swap {probs_swap.shape} differ")
    scores = _logp(probs) + _logp(probs_swap)[:, REV_PERM]
    scores[:, LABEL2ID[NEG]] += neg_bias
    return scores


def self_swap(
    df: pd.DataFrame, probs: np.ndarray, probs_swap: np.ndarray, neg_bias: float = 0.0
) -> list[str]:
    """Per-instance consistent decoding: the model is run on (B, A) by us, so no other
    test instance is looked up. Equals `joint` whenever the twin exists, up to float noise,
    and is reversal-consistent by construction (NEG is a fixed point of Rev)."""
    return [LABELS[i] for i in self_swap_scores(probs, probs_swap, neg_bias).argmax(axis=1)]


DECODERS = {
    "argmax": argmax,
    "self_swap": self_swap,
    "joint": joint,
    "joint_structural": joint_structural,
    "graph_soft": graph_soft,
    "graph_hard": graph_hard,
}
# Decoders whose output for a row depends only on that row's own text (PLAN.md 2c).
# Everything else looks up other instances of the test file and is analysis-only.
SAFE_DECODERS = ("argmax", "self_swap")
NEEDS_SWAP = ("self_swap",)


def decode(
    name: str,
    df: pd.DataFrame,
    probs: np.ndarray,
    config: dict | None = None,
    probs_swap: np.ndarray | None = None,
) -> list[str]:
    """Run a named decoder, passing graph-specific settings from an experiment config."""
    if name not in DECODERS:
        raise ValueError(f"unknown decoder {name!r}; choose from {list(DECODERS)}")
    if name == "self_swap":
        if probs_swap is None:
            raise ValueError("self_swap needs probs_swap = model probabilities on swap_texts(df)")
        return self_swap(df, probs, probs_swap, float((config or {}).get("neg_bias", 0.0)))
    if name not in {"graph_soft", "graph_hard"}:
        return DECODERS[name](df, probs)
    cfg = config or {}
    penalty = None if name == "graph_hard" else float(cfg.get("graph_lambda", 1.0))
    return graph_map(
        df,
        probs,
        transitivity_penalty=penalty,
        temperature=float(cfg.get("temperature", 1.0)),
        time_limit=cfg.get("graph_time_limit"),
    )


def correction_diagnostics(
    df: pd.DataFrame, before: list[str], after: list[str]
) -> dict[str, int | float | None]:
    """Count flips and, when gold labels exist, helpful/harmful corrections."""
    before_a, after_a = np.asarray(before), np.asarray(after)
    flipped = before_a != after_a
    result: dict[str, int | float | None] = {"flips": int(flipped.sum())}
    if "label" not in df:
        return result
    gold = df["label"].to_numpy()
    corrected = flipped & (before_a != gold) & (after_a == gold)
    harmed = flipped & (before_a == gold) & (after_a != gold)
    result.update(
        corrected=int(corrected.sum()),
        harmed=int(harmed.sum()),
        net_correction=int(corrected.sum() - harmed.sum()),
        flip_precision=float(corrected.sum() / flipped.sum()) if flipped.any() else None,
    )
    return result

"""Turn class probabilities into labels.

All decoders take `df` (with `rev_idx`, see data.add_reverse_index) and `probs`
of shape (n, 4) aligned with df rows, and return a list of label strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABEL2ID, LABELS, NEG, REV_PERM


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


DECODERS = {"argmax": argmax, "joint": joint, "joint_structural": joint_structural}

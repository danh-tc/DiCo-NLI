"""Product-of-experts debiasing against the surface (lexical overlap / length) shortcut.

A small bias model (logistic regression on the 12 surface features of `baselines.features`)
is fit per track. Its log-probabilities on the training rows are cross-fitted (5 folds grouped
by pair_id), so the bias model never scores a row it was fit on. During training the main
model is supervised through log_softmax(logits) + log p_bias (Clark et al. 2019; He et al.
2019); at test time only the main model is used, so predictions stay per-instance (PLAN 2c).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import features
from .data import LABEL2ID, LABELS


def _fit(x: np.ndarray, y: np.ndarray):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x, y)


def _logp(clf, x: np.ndarray) -> np.ndarray:
    p = np.full((len(x), len(LABELS)), 1e-6)
    p[:, clf[-1].classes_] = clf.predict_proba(x)
    p /= p.sum(1, keepdims=True)
    return np.log(p)


def surface_bias_logprobs(df: pd.DataFrame, folds: int = 5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cross-fitted log p_bias(.|A,B) and log p_bias(.|B,A) for every row of `df`."""
    df = df.reset_index(drop=True)
    swapped = df.assign(text1=df.text2, text2=df.text1)
    x, x_rev = features(df), features(swapped)
    y = df.label.map(LABEL2ID).to_numpy()
    lp, lp_rev = np.zeros((len(df), len(LABELS))), np.zeros((len(df), len(LABELS)))
    for track in sorted(df.track.unique()):
        rows = np.flatnonzero(df.track.to_numpy() == track)
        groups = df.pair_id.to_numpy()[rows]
        for fit_i, held_i in GroupKFold(n_splits=folds).split(rows, groups=groups):
            fit, held = rows[fit_i], rows[held_i]
            clf = _fit(x[fit], y[fit])
            lp[held], lp_rev[held] = _logp(clf, x[held]), _logp(clf, x_rev[held])
    report = {
        "bias_weighted_f1": float(f1_score(y, lp.argmax(1), average="weighted")),
        "bias_accuracy": float((lp.argmax(1) == y).mean()),
        "bias_mean_max_prob": float(np.exp(lp).max(1).mean()),
    }
    return lp.astype(np.float32), lp_rev.astype(np.float32), report

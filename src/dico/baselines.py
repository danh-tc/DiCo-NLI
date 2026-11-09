"""Non-neural baselines (no GPU needed).

    python -m dico.baselines --tracks 1 2 3 4

- majority: most frequent train label.
- rule: surface rule (equal -> EQ, B tokens inside A -> FWD, A inside B -> BWD, else NEG).
- surface_lr: logistic regression on length/overlap features (N1 shortcut probe).
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import env
from .data import BWD, EQ, FWD, LABEL2ID, LABELS, NEG, add_reverse_index, load_split
from .decode import DECODERS
from .evaluate import predict_and_score, save_json

_TOKEN = re.compile(r"\w+", re.UNICODE)


def toks(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def rule_label(a: str, b: str) -> str:
    ta, tb = set(toks(a)), set(toks(b))
    if ta == tb:
        return EQ
    if tb and tb < ta:
        return FWD
    if ta and ta < tb:
        return BWD
    return NEG


def features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for a, b in zip(df.text1, df.text2):
        ta, tb = toks(a), toks(b)
        sa, sb = set(ta), set(tb)
        inter = len(sa & sb)
        rows.append([
            len(ta), len(tb), len(ta) - len(tb),
            len(a), len(b), len(a) - len(b),
            inter / max(len(sa), 1), inter / max(len(sb), 1),
            inter / max(len(sa | sb), 1),
            float(sa == sb), float(sb < sa), float(sa < sb),
        ])
    return np.asarray(rows, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    print(env.describe())

    out_root = env.runs_dir() / "baselines"
    results = []
    for track in args.tracks:
        train = load_split("train", [track])
        dev = add_reverse_index(load_split("dev", [track]))
        out = out_root / f"track{track}"

        majority = train.label.value_counts().idxmax()
        results.append(predict_and_score(dev, [majority] * len(dev), "dev", track, out, "majority"))

        rule = [rule_label(a, b) for a, b in zip(dev.text1, dev.text2)]
        results.append(predict_and_score(dev, rule, "dev", track, out, "rule"))

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(features(train), train.label.map(LABEL2ID))
        probs = np.zeros((len(dev), len(LABELS)))
        probs[:, clf.classes_] = clf.predict_proba(features(dev))
        for name, decoder in DECODERS.items():
            results.append(predict_and_score(dev, decoder(dev, probs), "dev", track, out, f"surface_lr_{name}"))

    save_json(results, out_root / "summary.json")
    print(pd.DataFrame(results).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

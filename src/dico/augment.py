"""Transitive (natural-logic) augmentation: new labeled pairs from phrases shared by two train pairs.

For rows (A, B, l1) and (B, C, l2) of the same track, the pair (A, C) gets compose(l1, l2):
    EQ o X = X o EQ = X           (X in EQ, FWD, BWD, NEG)
    FWD o FWD = FWD, BWD o BWD = BWD
Other combinations (FWD o BWD, NEG o NEG, ...) imply nothing and are skipped. One step only.
Both orientations come out naturally, since the data hold both (A, B) and (B, A).

Only the given rows are used, so inside an OOF fold nothing comes from the held-out pairs;
`exclude` drops derived pairs whose (text1, text2) is an evaluation row (held-out fold or dev).
"""

from __future__ import annotations

import pandas as pd

EQ, FWD, BWD, NEG = "EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"
COMPOSE = {(EQ, x): x for x in (EQ, FWD, BWD, NEG)} | {(x, EQ): x for x in (FWD, BWD, NEG)} | {
    (FWD, FWD): FWD, (BWD, BWD): BWD}


def text_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df.text1, df.text2))


def transitive_rows(df: pd.DataFrame, exclude: set[tuple[str, str]] = frozenset()) -> pd.DataFrame:
    """Derived rows (same columns as df), per track; existing and excluded pairs are skipped."""
    out = []
    for track, part in df.groupby("track", sort=True):
        lang = {}
        edges: dict[str, dict[str, str]] = {}
        known = text_pairs(part)
        for r in part.itertuples(index=False):
            edges.setdefault(r.text1, {})[r.text2] = r.label
            lang[r.text1], lang[r.text2] = r.text1_lang, r.text2_lang
        derived: dict[tuple[str, str], set[str]] = {}
        for a, nbrs in edges.items():
            for b, l1 in nbrs.items():
                for c, l2 in edges.get(b, {}).items():
                    lab = COMPOSE.get((l1, l2))
                    if lab is None or c == a or (a, c) in known or (a, c) in exclude:
                        continue
                    derived.setdefault((a, c), set()).add(lab)
        for k, ((a, c), labs) in enumerate(sorted(derived.items())):
            if len(labs) != 1:  # two paths disagree: drop
                continue
            out.append({"instance_id": f"aug_t{track}_{k:06d}", "pair_id": f"aug_t{track}_{k:06d}",
                        "text1_lang": lang[a], "text2_lang": lang[c], "text1": a, "text2": c,
                        "label": labs.pop(), "track": track})
    return pd.DataFrame(out, columns=list(df.columns)) if out else df.iloc[:0].copy()


def augment(df: pd.DataFrame, exclude: set[tuple[str, str]] = frozenset()) -> pd.DataFrame:
    extra = transitive_rows(df, exclude)
    print(f"augment_transitive: +{len(extra)} rows "
          f"({extra.label.value_counts().to_dict() if len(extra) else {}})")
    return pd.concat([df, extra], ignore_index=True)

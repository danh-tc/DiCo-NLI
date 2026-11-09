"""Data loading, label algebra and reverse-pair matching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import env

EQ, FWD, BWD, NEG = "EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"
LABELS = [EQ, FWD, BWD, NEG]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
REV = {EQ: EQ, FWD: BWD, BWD: FWD, NEG: NEG}
# REV as a permutation of label indices: probs_of_reverse[:, REV_PERM] aligns with this instance.
REV_PERM = [LABEL2ID[REV[label]] for label in LABELS]

TEXT_COLS = ["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2"]


def split_file(split: str, track: int, kind: str = "participant_labeled") -> Path:
    return env.data_dir() / split / f"dico_nli_{split}_track{track}_{kind}.csv"


def read_csv(path: str | Path) -> pd.DataFrame:
    # keep_default_na=False: phrases such as "NA" or "None" must stay strings.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def load_split(split: str, tracks: list[int]) -> pd.DataFrame:
    frames = []
    for track in tracks:
        df = read_csv(split_file(split, track))
        df["track"] = track
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_reverse_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add `rev_idx`: row index of the instance with swapped texts (-1 if absent).

    Matching is by text swap within a pair_id, not by the original/flipped suffix:
    in track 4, `en-es__flipped` reverses to `es-en__original`.
    """
    df = df.reset_index(drop=True).copy()
    key = list(zip(df.pair_id, df.text1_lang, df.text1, df.text2_lang, df.text2))
    rev_key = list(zip(df.pair_id, df.text2_lang, df.text2, df.text1_lang, df.text1))
    pos = {k: i for i, k in enumerate(key)}
    df["rev_idx"] = [pos.get(k, -1) for k in rev_key]
    return df

"""Score predictions with the organizers' official scorer (imported, not reimplemented)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from . import env
from .data import split_file

MAIN_METRICS = ["weighted_f1", "soft_cons", "hard_cons", "macro_f1", "accuracy"]


def _scorer():
    repo = str(env.task_repo_dir())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from evaluation_functions.scorer import score_files  # noqa: E402

    return score_files


def write_predictions(df: pd.DataFrame, labels: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"instance_id": df["instance_id"], "label": labels}).to_csv(path, index=False)
    return path


def score(pred_path: Path, split: str, track: int, out_dir: Path) -> dict:
    """Run the official scorer; writes scores.json/scores.txt to out_dir, returns main metrics."""
    report = _scorer()(split_file(split, track, "reference"), pred_path, output_dir=out_dir)
    full = report.to_dict()
    return {k: _find(full, k) for k in MAIN_METRICS}


def _find(d, key):
    """Find a metric by name anywhere in the (nested) report dict."""
    if isinstance(d, dict):
        if key in d and not isinstance(d[key], (dict, list)):
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def predict_and_score(df, labels, split, track, out_dir: Path, name: str) -> dict:
    pred_path = write_predictions(df, labels, out_dir / f"{split}_track{track}_{name}.csv")
    metrics = score(pred_path, split, track, out_dir / f"scores_{split}_track{track}_{name}")
    return {"split": split, "track": track, "decoder": name, **metrics}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

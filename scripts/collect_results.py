"""Aggregate runs/*/seed*/results.json into results/summary.csv (mean ± std over seeds).

    python scripts/collect_results.py [--runs_dir runs]

Copy runs/ from Kaggle into the local runs/ folder first, then all results land in one table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dico import env  # noqa: E402

METRICS = ["weighted_f1", "soft_cons", "hard_cons"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(env.runs_dir()))
    args = ap.parse_args()

    rows = []
    for path in Path(args.runs_dir).glob("*/seed*/results.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data["results"]:
            rows.append({**r, "train_minutes": data.get("train_minutes")})
    if not rows:
        print(f"no results under {args.runs_dir}")
        return

    df = pd.DataFrame(rows)
    keys = ["name", "track", "decoder"]
    agg = df.groupby(keys)[METRICS].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = df.groupby(keys).seed.nunique()
    agg = agg.reset_index().sort_values(["track", "name", "decoder"])

    out = ROOT / "results" / "summary.csv"
    out.parent.mkdir(exist_ok=True)
    agg.to_csv(out, index=False)
    df.to_csv(ROOT / "results" / "all_runs.csv", index=False)
    print(agg.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

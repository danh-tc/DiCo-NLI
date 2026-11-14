"""Audit whether DiCo-NLI splits contain usable observed graph constraints.

Reports exact-phrase graph structure plus reversal/composition violations in
gold labels. It never creates latent edges or merges phrases across languages.

    python scripts/audit_graph.py --splits train dev --tracks 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.data import REV, add_reverse_index, load_split  # noqa: E402
from dico.graph_decode import COMPOSITION, _component_rows, _observed_paths  # noqa: E402


def audit(split: str, track: int) -> dict:
    df = add_reverse_index(load_split(split, [track]))
    components = _component_rows(df)
    paths = [path for rows in components for path in _observed_paths(df, rows)]
    reverse_pairs = [(i, int(j)) for i, j in enumerate(df.rev_idx) if int(j) >= 0 and i < int(j)]
    reversal_violations = sum(
        df.iloc[j].label != REV[df.iloc[i].label] for i, j in reverse_pairs
    )
    applicable = violations = 0
    for i, j, k in paths:
        expected = COMPOSITION.get((df.iloc[i].label, df.iloc[j].label))
        if expected is not None:
            applicable += 1
            violations += df.iloc[k].label != expected
    return {
        "split": split,
        "track": track,
        "rows": len(df),
        "nodes": len(set(zip(df.text1_lang, df.text1)) | set(zip(df.text2_lang, df.text2))),
        "components": len(components),
        "largest_component_edges": max(map(len, components), default=0),
        "reverse_pairs": len(reverse_pairs),
        "reversal_gold_violations": reversal_violations,
        "observed_paths": len(paths),
        "applicable_gold_paths": applicable,
        "composition_gold_violations": violations,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    rows = [audit(split, track) for split in args.splits for track in args.tracks]
    try:
        import pandas as pd

        print(pd.DataFrame(rows).to_string(index=False))
    except ImportError:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()

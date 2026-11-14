"""Tune/evaluate post-hoc graph decoding on a saved dev probability file.

This is a parameter-selection utility for a later test submission.  Scores from
the same dev set used to fit temperature/select lambda are exploratory and must
not be presented as held-out estimates.

Example:
    python scripts/graph_ablation.py --run runs/m0_deberta_base_en/seed1 --track 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.data import LABEL2ID, add_reverse_index, load_split  # noqa: E402
from dico.decode import argmax, correction_diagnostics  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402
from dico.graph_decode import fit_temperature, graph_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--temperature", type=float, help="fixed T; default fits T on this dev set")
    ap.add_argument("--time_limit", type=float, default=30.0)
    args = ap.parse_args()

    df = add_reverse_index(load_split("dev", [args.track]))
    probs_path = args.run / f"dev_track{args.track}_probs.npy"
    probs = np.load(probs_path)
    gold = df.label.map(LABEL2ID).to_numpy()
    temperature = args.temperature or fit_temperature(probs, gold)
    raw = argmax(df, probs)
    out_dir = args.run / f"graph_tuning_track{args.track}"
    rows = []

    settings = [(f"graph_soft_lam{lam:g}", lam) for lam in args.lambdas]
    settings.append(("graph_hard", None))
    for name, penalty in settings:
        labels, stats = graph_map(
            df,
            probs,
            transitivity_penalty=penalty,
            temperature=temperature,
            time_limit=args.time_limit,
            return_stats=True,
        )
        row = predict_and_score(df, labels, "dev", args.track, out_dir, name)
        row.update(correction_diagnostics(df, raw, labels))
        row.update(stats.__dict__)
        row.update(name=name, temperature=temperature, graph_lambda=penalty)
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "ablation.csv", index=False)
    best = results.iloc[0]
    params = {
        "temperature": float(temperature),
        "decoder": str(best["name"]),
        "graph_lambda": None if pd.isna(best["graph_lambda"]) else float(best["graph_lambda"]),
        "selection_split": "dev",
        "warning": "dev-tuned; use only for later test inference, not as held-out evidence",
    }
    (out_dir / "best_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"temperature={temperature:.4f} | observed_paths={rows[0]['observed_paths']}")
    print(results[[
        "name", "weighted_f1", "soft_cons", "hard_cons", "flips",
        "corrected", "harmed", "net_correction", "flip_precision",
    ]].round(4).to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

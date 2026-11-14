"""Compare runs on per-track scores and cross-language agreement on dev.

Agreement: for each dev instance, do the EN, ES and EU predictions of the SAME instance
(aligned by instance_id without the language tag) coincide? Uses each run's saved
`dev_track{1,2,3}_<decoder>.csv`, so nothing is re-run.

    python scripts/xling_agreement.py runs/quick_base runs/quick_js runs/quick_en
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dico.data import load_split  # noqa: E402

TRACKS = {1: "en", 2: "es", 3: "eu"}


def key(ids: pd.Series) -> pd.Series:
    return ids.str.replace(r"__[a-z]{2}-[a-z]{2}__", "__", regex=True)


def agreement(seed_dir: Path, decoder: str) -> dict:
    preds = {}
    for t, lang in TRACKS.items():
        path = seed_dir / f"dev_track{t}_{decoder}.csv"
        if not path.exists():
            return {}
        df = load_split("dev", [t])
        p = pd.read_csv(path, keep_default_na=False).rename(columns={"label": "pred"})
        m = df.merge(p, on="instance_id").assign(k=lambda x: key(x.instance_id)).set_index("k")
        preds[lang] = m
    idx = preds["en"].index
    gold = preds["en"].label
    en, es, eu = (preds[x].pred.loc[idx] for x in ("en", "es", "eu"))
    agree = (en == es) & (es == eu)
    return {
        "disagree": round(float(1 - agree.mean()), 4),
        "acc_if_agree": round(float((en[agree] == gold[agree]).mean()), 4),
        "eu_wrong_other_right": round(float(((eu != gold) & ((en == gold) | (es == gold))).mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--decoder", default="self_swap")
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        for seed_dir in sorted(run.glob("seed*")):
            res = json.loads((seed_dir / "results.json").read_text(encoding="utf-8"))
            scores = {r["track"]: r for r in res["results"] if r["decoder"] == args.decoder}
            row = {"run": run.name, "seed": seed_dir.name}
            for t, lang in TRACKS.items():
                if t in scores:
                    r = scores[t]
                    row[lang] = f"{r['weighted_f1']:.3f}/{r['soft_cons']:.3f}/{r['hard_cons']:.3f}"
            row.update(agreement(seed_dir, args.decoder))
            rows.append(row)
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

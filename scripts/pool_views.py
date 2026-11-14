"""Cross-track source pooling: decode each source pair_id ONCE from all its views.

Every pair_id appears in all 4 tracks (en/es/eu mono + 6 mixed combos, x2 directions) and the
gold label is identical across languages (checked on dev: 100%). Orientation comes from the
instance_id suffix: `original` = (phrase1, phrase2), `flipped` = (phrase2, phrase1), so a
flipped view's distribution is mapped through Rev before summing log-probs.

EXPERIMENTAL / test legality must be confirmed with the organizers (uses other test instances).

    python scripts/pool_views.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dico import evaluate  # noqa: E402
from dico.data import LABELS, NEG, LABEL2ID, REV_PERM, load_split  # noqa: E402

# (run dir, tracks it has probs for, weight)
SOURCES = {
    "mdeberta": [("runs/m0_mdeberta_base_all", [1, 2, 3, 4], 1.0)],
    "mdeberta+debertaEN": [("runs/m0_mdeberta_base_all", [1, 2, 3, 4], 1.0),
                           ("runs/m0_deberta_base_en", [1], 1.0)],
}


def collect(sources) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    dfs = {t: load_split("dev", [t]).reset_index(drop=True) for t in (1, 2, 3, 4)}
    parts = []
    for run, tracks, w in sources:
        for sd in sorted((ROOT / run).glob("seed*")):
            for t in tracks:
                f = sd / f"dev_track{t}_probs.npy"
                if not f.exists():
                    continue
                lp = np.log(np.clip(np.load(f), 1e-12, 1.0))
                df = dfs[t]
                flipped = df.instance_id.str.endswith("__flipped").to_numpy()
                lp[flipped] = lp[flipped][:, REV_PERM]  # map to original orientation
                parts.append(pd.DataFrame(w * lp, columns=LABELS).assign(pair_id=df.pair_id.values))
    pooled = pd.concat(parts).groupby("pair_id")[LABELS].mean()
    return dfs, pooled


def decode(df: pd.DataFrame, pooled: pd.DataFrame) -> list[str]:
    lab = pooled.loc[df.pair_id].to_numpy().argmax(axis=1)
    flipped = df.instance_id.str.endswith("__flipped").to_numpy()
    lab = np.where(flipped, np.asarray(REV_PERM)[lab], lab)
    return [LABELS[i] for i in lab]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/pool_views")
    args = ap.parse_args()
    rows = []
    for name, sources in SOURCES.items():
        dfs, pooled = collect(sources)
        for t, df in dfs.items():
            labels = decode(df, pooled)
            r = evaluate.predict_and_score(df, labels, "dev", t, args.out / name, f"dev_track{t}_pooled")
            rows.append({"sources": name, "track": t, **{k: round(r[k], 4) for k in ("weighted_f1", "soft_cons", "hard_cons")}})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\nmacro4:\n", res.groupby("sources")[["weighted_f1", "soft_cons", "hard_cons"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()

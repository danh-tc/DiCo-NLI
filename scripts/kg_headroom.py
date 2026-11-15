"""WordNet headroom for the main system, measured on train OOF (dev not used).

For every train row, kg_pilot.bucket() diffs the content words of A and B and, for substitutions,
looks up the strongest WordNet relation (syn / hypo / hyper / ant / cohypo -> an implied label).
The main-system stack (configs/main_system.yaml) is decoded on the train OOF probabilities.

  headroom = stack wrong and the WordNet label right   (what a WordNet signal could fix)
  risk     = stack right and the WordNet label wrong   (what following WordNet would break)

    python scripts/kg_headroom.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dico.data import load_split  # noqa: E402
from kg_pilot import REL_LABEL, bucket  # noqa: E402
from stack_qwen_mdeberta import decode, oof_scores  # noqa: E402


def main() -> None:
    system = yaml.safe_load((ROOT / "configs" / "main_system.yaml").read_text(encoding="utf-8"))
    out = ROOT / "runs" / "kg_headroom"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, s in system["tracks"].items():
        train = load_split("train", [int(t)]).reset_index(drop=True)
        pred = decode(oof_scores(s["qwen"], train), oof_scores(system["mdeberta"], train),
                      s["w_qwen"], s["neg_bias"])
        b = train.apply(bucket, axis=1, result_type="expand")
        train["cat"] = np.where(b[0] == "subst", "subst:" + b[1], b[0])
        train["wn"] = b[1].map(REL_LABEL)
        train["pred"] = pred
        train["err"] = train.pred != train.label
        train.drop(columns=["text1", "text2"]).to_csv(out / f"train_track{t}.csv", index=False)
        has = train.wn.notna()
        n_err = int(train.err.sum())
        head = int((has & train.err & (train.wn == train.label)).sum())
        risk = int((has & ~train.err & (train.wn != train.label)).sum())
        rows.append({
            "track": int(t), "n": len(train), "stack_errors": n_err,
            "with_wn_relation": int(has.sum()),
            "wn_rule_acc": round(float((train.wn == train.label)[has].mean()), 3) if has.any() else None,
            "headroom": head, "headroom_%err": round(100 * head / n_err, 1),
            "risk": risk, "go": head >= 0.05 * n_err and head > risk,
        })
        print(f"\n== track {t}: error rate by category (train OOF, main-system stack)")
        tab = train.groupby("cat").agg(n=("err", "size"), err_rate=("err", "mean"), errors=("err", "sum"))
        tab["share_of_errors"] = (tab.errors / n_err).round(3)
        print(tab.round(3).sort_values("errors", ascending=False).to_string())
    print("\n== headroom (pre-registered go: headroom >= 5% of errors and headroom > risk)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

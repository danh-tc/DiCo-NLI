"""Export M0 errors (dev track 1) for blind two-axis human annotation.

One row per source pair (the `original` direction), sampled among pairs where the M0 joint
decoder is wrong in at least one seed; stratified so ~2/3 come from the `subst:none` bucket
(paraphrase / world knowledge, the group under study) and the rest from other buckets.

Writes annotation/ :
  1_blind.csv   annotate FIRST, without looking at 2_reveal.csv:
                my_label = your own NLI label for (text1 -> text2)
  2_reveal.csv  then fill, per row:
                gold_reliability = clear | ambiguous | likely_incorrect | insufficient_context
                capabilities     = any of lexical;compositional;world_knowledge;pragmatic;other
                (semicolon-separated, multi-label)

    python scripts/export_annotation.py --n 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kg_pilot import bucket  # noqa: E402
from dico.data import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--run", type=Path, default=ROOT / "runs/m0_deberta_base_en")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    df = load_split("dev", [1]).reset_index(drop=True)
    preds = {}
    for sd in sorted(args.run.glob("seed*")):
        p = pd.read_csv(sd / "dev_track1_joint.csv", keep_default_na=False).set_index("instance_id").label
        preds[sd.name] = df.instance_id.map(p)
    P = pd.DataFrame(preds)
    df["m0_preds"] = P.apply(lambda r: "/".join(r), axis=1)
    df["n_wrong"] = P.ne(df.label, axis=0).sum(axis=1)

    pair_wrong = df.groupby("pair_id").n_wrong.max()
    orig = df[df.instance_id.str.endswith("__original") & df.pair_id.map(pair_wrong).gt(0)].copy()
    b = orig.apply(bucket, axis=1, result_type="expand")
    orig["category"] = b[0].where(b[0] != "subst", "subst:" + b[1])

    target = orig[orig.category == "subst:none"]
    rest = orig[orig.category != "subst:none"]
    k = min(len(target), round(args.n * 2 / 3))
    sample = pd.concat([target.sample(k, random_state=args.seed),
                        rest.sample(min(len(rest), args.n - k), random_state=args.seed)])
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle buckets
    sample.insert(0, "uid", [f"a{i:03d}" for i in range(len(sample))])

    out = ROOT / "annotation"
    out.mkdir(exist_ok=True)
    blind = sample[["uid", "text1", "text2"]].assign(my_label="", notes="")
    reveal = sample[["uid", "text1", "text2", "label", "m0_preds", "n_wrong", "category", "pair_id"]].rename(
        columns={"label": "gold"}).assign(gold_reliability="", capabilities="", notes="")
    blind.to_csv(out / "1_blind.csv", index=False, encoding="utf-8-sig")  # BOM: opens cleanly in Excel
    reveal.to_csv(out / "2_reveal.csv", index=False, encoding="utf-8-sig")
    print(f"{len(sample)} pairs ({k} subst:none, {len(sample) - k} other) of {len(orig)} erroneous pairs")
    print(sample.category.value_counts().to_string())
    print(f"-> {out / '1_blind.csv'}\n-> {out / '2_reveal.csv'}")


if __name__ == "__main__":
    main()

"""Choose the self_swap NEG bias on train OOF, then apply it unchanged to dev (held-out).

The bias is picked from grouped train OOF probabilities written by generate_oof.py
(train_oof_probs.npy + train_oof_swap_probs.npy). Dev labels are never used for the
choice, so the dev scores printed at the end are an honest estimate (PLAN.md 2c).

    python scripts/tune_neg_bias.py --config configs/m0_mdeberta_xnli_pair_all.yaml \
        --runs runs/m0_mdeberta_xnli_pair_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.data import NEG, REV, add_reverse_index, load_split  # noqa: E402
from dico.decode import self_swap  # noqa: E402
from dico.evaluate import predict_and_score, save_json  # noqa: E402
from dico.train import load_config  # noqa: E402


def local_metrics(df: pd.DataFrame, pred: list[str]) -> dict[str, float]:
    """Weighted F1, SoftCons, HardCons as the official scorer defines them.

    Cons metrics run over gold reversible pairs (rows whose swapped twin is in the file);
    a NEG prediction on such a pair is never consistent. The twin lookup here is only
    for *scoring* labelled train data; the decoder itself never uses it.
    """
    gold = df["label"].to_numpy()
    pred_a = np.asarray(pred)
    rev = df["rev_idx"].to_numpy()
    pairs = [(i, j) for i, j in enumerate(rev) if j > i]
    soft = [pred_a[j] == REV[pred_a[i]] and pred_a[i] != NEG for i, j in pairs]
    hard = [pred_a[i] == gold[i] and pred_a[j] == gold[j] for i, j in pairs]
    return {
        "weighted_f1": float(f1_score(gold, pred_a, average="weighted")),
        "soft_cons": float(np.mean(soft)) if pairs else float("nan"),
        "hard_cons": float(np.mean(hard)) if pairs else float("nan"),
    }


def objective(m: dict[str, float]) -> float:
    return (m["weighted_f1"] + m["soft_cons"] + m["hard_cons"]) / 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oof", type=Path, help="default: runs/oof/<config name>")
    ap.add_argument("--runs", type=Path, help="run dir with seed*/dev_track*_{probs,swap_probs}.npy")
    ap.add_argument("--grid", type=float, nargs=3, default=[-20.0, 2.0, 0.5], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--fixed", type=float,
                    help="use this bias (picked by hand from the OOF curve, before looking at dev)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, as in dico.train")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    oof_dir = args.oof or ROOT / "runs" / "oof" / cfg["name"]
    train = load_split("train", cfg["train_tracks"]).reset_index(drop=True)
    rows = pd.read_csv(oof_dir / "train_oof_rows.csv", dtype=str, keep_default_na=False)
    if not rows.instance_id.equals(train.instance_id):
        raise SystemExit("train_oof_rows.csv does not match the train split order")
    probs = np.load(oof_dir / "train_oof_probs.npy")
    probs_swap = np.load(oof_dir / "train_oof_swap_probs.npy")

    # Score per track (the official metrics are per track), choose by the mean over tracks.
    tracks = {t: add_reverse_index(train[train.track == t]) for t in cfg["train_tracks"]}
    index = {t: np.flatnonzero(train.track.to_numpy() == t) for t in tracks}
    lo, hi, step = args.grid
    grid = np.round(np.arange(lo, hi + step / 2, step), 6)
    table = []
    for bias in grid:
        per = {t: local_metrics(df, self_swap(df, probs[index[t]], probs_swap[index[t]], bias))
               for t, df in tracks.items()}
        row = {"neg_bias": float(bias)}
        for key in ("weighted_f1", "soft_cons", "hard_cons"):
            row[key] = float(np.mean([m[key] for m in per.values()]))
        row["objective"] = objective(row)
        table.append(row)
    table = pd.DataFrame(table)
    best = table.loc[table.objective.idxmax()]
    zero = table.loc[(table.neg_bias - 0.0).abs().idxmin()]
    table.to_csv(oof_dir / "neg_bias_grid.csv", index=False)
    if args.fixed is None:
        chosen, rule = best, "argmax of mean(F1, SoftCons, HardCons) on train OOF"
    else:
        chosen = table.loc[(table.neg_bias - args.fixed).abs().idxmin()]
        rule = (f"fixed at {args.fixed} from the train-OOF curve (near-best objective, "
                "further from the NEG-collapse cliff); dev not consulted")
    choice = {"neg_bias": float(chosen.neg_bias), "train_oof_at_choice": chosen.to_dict(),
              "train_oof_at_best": best.to_dict(), "train_oof_at_zero": zero.to_dict(),
              "selected_on": "train OOF (grouped by pair_id)", "rule": rule}
    save_json(choice, oof_dir / "neg_bias.json")
    print("train OOF (mean over tracks)")
    print(pd.DataFrame([zero, best, chosen], index=["bias=0", "best", "chosen"]).round(4).to_string())

    if not args.runs:
        return
    results = []
    for seed_dir in sorted(args.runs.glob("seed*")):
        for t in cfg["eval_tracks"]:
            p_path, s_path = seed_dir / f"dev_track{t}_probs.npy", seed_dir / f"dev_track{t}_swap_probs.npy"
            if not s_path.exists():
                print(f"skip {seed_dir.name} track{t}: no swap probs (re-run training)")
                continue
            df = load_split("dev", [t]).reset_index(drop=True)
            labels = self_swap(df, np.load(p_path), np.load(s_path), choice["neg_bias"])
            row = predict_and_score(df, labels, "dev", t, seed_dir, "self_swap_negbias")
            results.append({"seed": seed_dir.name, "neg_bias": choice["neg_bias"], **row})
    if results:
        out = pd.DataFrame(results)
        out.to_csv(args.runs / "self_swap_negbias_dev.csv", index=False)
        cols = ["weighted_f1", "soft_cons", "hard_cons"]
        print("\ndev (held-out), self_swap + train-OOF neg_bias")
        print(out.groupby("track")[cols].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()

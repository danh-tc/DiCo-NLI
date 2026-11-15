"""Stack Qwen (one model per track) with the mDeBERTa main system, per track (PLAN.md 2d, step 4).

Per track t in EN/ES/EU, each model contributes its self_swap score
    S(l) = log p(l|A,B) + log p(Rev(l)|B,A)
and the stack decodes argmax_l  w * S_qwen(l) + (1 - w) * S_mdeb(l) + neg_bias * [l = NEG].
Only the row itself and its swap are used, so the stack is submission-safe like self_swap.

(w, neg_bias) are chosen per track on train OOF (both models' OOF, grouped by pair_id),
with the two pre-registered objectives of tune_neg_bias.py; dev is only reported.
Dev uses the 3-seed ensemble (mean probabilities) of each model. Baselines: each model
alone at its own train-OOF bias (w = 1 / w = 0 in the same grid).

    python scripts/stack_qwen_mdeberta.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from dico.data import LABEL2ID, LABELS, NEG, add_reverse_index, load_split  # noqa: E402
from dico.decode import self_swap_scores  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402
from tune_neg_bias import OBJECTIVES, local_metrics  # noqa: E402

MDEB = "m0_mdeberta_xnli_pair_all"
LANG = {1: "en", 2: "es", 3: "eu", 4: "mixed"}
METRICS = ("weighted_f1", "soft_cons", "hard_cons")


def oof_scores(name: str, train: pd.DataFrame) -> np.ndarray:
    """Self_swap scores (bias 0) of a model's train OOF, aligned to `train` rows."""
    d = ROOT / "runs" / "oof" / name
    rows = pd.read_csv(d / "train_oof_rows.csv", dtype=str, keep_default_na=False)
    pos = pd.Series(np.arange(len(rows)), index=rows.track + "|" + rows.instance_id)
    idx = pos.loc[train.track.astype(str) + "|" + train.instance_id].to_numpy()
    p, s = np.load(d / "train_oof_probs.npy")[idx], np.load(d / "train_oof_swap_probs.npy")[idx]
    return self_swap_scores(p, s)


def dev_scores(name: str, track: int) -> np.ndarray:
    seeds = sorted((ROOT / "runs" / name).glob("seed*"))
    p = np.mean([np.load(s / f"dev_track{track}_probs.npy") for s in seeds], 0)
    q = np.mean([np.load(s / f"dev_track{track}_swap_probs.npy") for s in seeds], 0)
    return self_swap_scores(p, q)


def decode(sq: np.ndarray, sm: np.ndarray, w: float, bias: float) -> list[str]:
    s = w * sq + (1 - w) * sm
    s[:, LABEL2ID[NEG]] += bias
    return [LABELS[i] for i in s.argmax(1)]


def bootstrap(df: pd.DataFrame, a: list[str], b: list[str], draws: int, seed: int) -> dict:
    """Paired bootstrap over source pairs (pair_id) of metric(a) - metric(b)."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("pair_id").indices
    keys = list(groups)
    rev = df.rev_idx.to_numpy()
    gold = df.label.to_numpy()
    a, b = np.asarray(a), np.asarray(b)
    out = {m: [] for m in METRICS}
    for _ in range(draws):
        idx, new_rev = [], []
        for k in rng.integers(0, len(keys), len(keys)):
            g = groups[keys[k]]  # reverse twins always share a pair_id
            local = {int(i): len(idx) + n for n, i in enumerate(g)}
            new_rev += [local.get(int(rev[i]), -1) for i in g]
            idx += list(g)
        sub = pd.DataFrame({"label": gold[idx], "rev_idx": new_rev})
        ma = local_metrics(sub, list(a[idx]))
        mb = local_metrics(sub, list(b[idx]))
        for m in METRICS:
            out[m].append(ma[m] - mb[m])
    return {m: {"mean": float(np.mean(v)), "ci95": [float(x) for x in np.quantile(v, [0.025, 0.975])],
                "p_gt0": float(np.mean(np.asarray(v) > 0))} for m, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_step", type=float, default=0.05)
    ap.add_argument("--grid", type=float, nargs=3, default=[-20.0, 2.0, 0.5], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--qwen", default="qwen_cls_{lang}",
                    help="Qwen run name; {lang} = en/es/eu/mixed (e.g. qwen_cls_all for the 4-track model)")
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "stack_qwen_mdeberta")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ws = np.round(np.arange(0, 1 + args.w_step / 2, args.w_step), 6)
    lo, hi, step = args.grid
    biases = np.round(np.arange(lo, hi + step / 2, step), 6)

    report, choices, lines = [], {}, []
    for t in args.tracks:
        lang = LANG[t]
        qwen = args.qwen.format(lang=lang)
        train = add_reverse_index(load_split("train", [t]).reset_index(drop=True))
        sq, sm = oof_scores(qwen, train), oof_scores(MDEB, train)
        grid = []
        for w in ws:
            for b in biases:
                m = local_metrics(train, decode(sq, sm, w, b))
                grid.append({"w_qwen": float(w), "neg_bias": float(b), **m})
        grid = pd.DataFrame(grid)
        grid.to_csv(args.out / f"train_oof_grid_track{t}.csv", index=False)

        dev = add_reverse_index(load_split("dev", [t]).reset_index(drop=True))
        dq, dm = dev_scores(qwen, t), dev_scores(MDEB, t)
        for obj, (fn, rule, _) in OBJECTIVES.items():
            grid["objective"] = grid.apply(fn, axis=1)
            picks = {
                "stack": grid.loc[grid.objective.idxmax()],
                "qwen": grid[grid.w_qwen == 1.0].pipe(lambda g: g.loc[g.objective.idxmax()]),
                "mdeb": grid[grid.w_qwen == 0.0].pipe(lambda g: g.loc[g.objective.idxmax()]),
            }
            preds = {}
            for system, row in picks.items():
                preds[system] = decode(dq, dm, row.w_qwen, row.neg_bias)
                m = local_metrics(dev, preds[system])
                report.append({"track": lang, "objective": obj, "system": system,
                               "w_qwen": row.w_qwen, "neg_bias": row.neg_bias,
                               "oof_objective": round(row.objective, 4),
                               **{f"dev_{k}": round(v, 4) for k, v in m.items()}})
            choices[f"track{t}_{obj}"] = {"w_qwen": float(picks["stack"].w_qwen),
                                          "neg_bias": float(picks["stack"].neg_bias),
                                          "rule": f"{rule}, grid over (w_qwen, neg_bias)"}
            if args.bootstrap:
                bs = bootstrap(dev, preds["stack"], preds["qwen"], args.bootstrap, args.seed)
                choices[f"track{t}_{obj}"]["dev_bootstrap_stack_minus_qwen"] = bs
                lines.append(f"{lang} {obj}: stack - qwen  " + "  ".join(
                    f"{m} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] P>0={v['p_gt0']:.2f}"
                    for m, v in bs.items()))

            if obj == "mean3":  # main system (PLAN.md 2d): write + score with the official scorer
                predict_and_score(dev, preds["stack"], "dev", t, args.out, "main")

    # Track 4 (Mixed) without a Qwen model: mDeBERTa alone at its train-OOF bias (mean3).
    if 4 not in args.tracks:
        mdeb_bias = json.loads((ROOT / "runs" / "oof" / MDEB / "neg_bias.json").read_text())["neg_bias"]
        dev4 = add_reverse_index(load_split("dev", [4]).reset_index(drop=True))
        s4 = dev_scores(MDEB, 4)
        predict_and_score(dev4, decode(s4, s4, 0.0, mdeb_bias), "dev", 4, args.out, "main")

    table = pd.DataFrame(report)
    table.to_csv(args.out / "summary.csv", index=False)
    (args.out / "choices.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print("\npaired bootstrap over pair_id on dev (stack vs qwen, same objective)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

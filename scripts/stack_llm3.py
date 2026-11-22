"""Idea B (PLAN.md 2d): add the zero-shot LLM expert as a third model to the per-track stack.

    argmax_l  w_q*S_qwen(l) + w_m*S_mdeb(l) + w_l*S_llm(l) + neg_bias*[l = NEG]

S = self_swap score of each model (submission-safe). (w_q, w_m, w_l) on the simplex with step
--w_step and neg_bias are chosen per track on train OOF (objective mean3); the LLM is not
trained, so its train probabilities stand in for OOF. Rule: keep the 3-model stack only if
its OOF objective >= the 2-model stack (w_l = 0 in the same grid) + 0.003. Dev is reported.

    python scripts/stack_llm3.py --llm llm7b_expert
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
from dico.evaluate import predict_and_score  # noqa: E402
from stack_qwen_mdeberta import LANG, MDEB, bootstrap, dev_scores, oof_scores  # noqa: E402
from tune_neg_bias import OBJECTIVES, local_metrics  # noqa: E402

MARGIN = 0.003


def decode(scores: list[np.ndarray], w: tuple[float, ...], bias: float) -> list[str]:
    s = sum(wi * si for wi, si in zip(w, scores))
    s = s.copy()
    s[:, LABEL2ID[NEG]] += bias
    return [LABELS[i] for i in s.argmax(1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="llm7b_expert")
    ap.add_argument("--qwen", default="qwen_cls_{lang}")
    ap.add_argument("--mdeb", default=MDEB)
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--w_step", type=float, default=0.05)
    ap.add_argument("--grid", type=float, nargs=3, default=[-20.0, 2.0, 0.5], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "stack_llm3")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = int(round(1 / args.w_step))
    weights = [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    lo, hi, step = args.grid
    biases = np.round(np.arange(lo, hi + step / 2, step), 6)
    fn = OBJECTIVES["mean3"][0]

    report, choices = [], {}
    for t in args.tracks:
        lang = LANG[t]
        names = (args.qwen.format(lang=lang), args.mdeb, args.llm)
        train = add_reverse_index(load_split("train", [t]).reset_index(drop=True))
        dev = add_reverse_index(load_split("dev", [t]).reset_index(drop=True))
        tr = [oof_scores(m, train) for m in names]
        dv = [dev_scores(m, t) for m in names]
        grid = []
        for w in weights:
            for b in biases:
                m = local_metrics(train, decode(tr, w, b))
                grid.append({"w_qwen": w[0], "w_mdeb": w[1], "w_llm": w[2], "neg_bias": float(b),
                             **m, "objective": fn(m)})
        grid = pd.DataFrame(grid)
        grid.to_csv(args.out / f"train_oof_grid_track{t}.csv", index=False)
        picks = {
            "stack3": grid.loc[grid.objective.idxmax()],
            "stack2": grid[grid.w_llm == 0].pipe(lambda g: g.loc[g.objective.idxmax()]),
            "llm": grid[grid.w_llm == 1].pipe(lambda g: g.loc[g.objective.idxmax()]),
        }
        preds = {}
        for system, row in picks.items():
            w = (row.w_qwen, row.w_mdeb, row.w_llm)
            preds[system] = decode(dv, w, row.neg_bias)
            m = local_metrics(dev, preds[system])
            report.append({"track": lang, "system": system, "w_qwen": row.w_qwen, "w_mdeb": row.w_mdeb,
                           "w_llm": row.w_llm, "neg_bias": row.neg_bias,
                           "oof_objective": round(row.objective, 4),
                           **{f"dev_{k}": round(v, 4) for k, v in m.items()}})
        gain = float(picks["stack3"].objective - picks["stack2"].objective)
        choices[f"track{t}"] = {
            "go": gain >= MARGIN, "oof_gain": round(gain, 4), "margin": MARGIN,
            **{k: float(picks["stack3"][k]) for k in ("w_qwen", "w_mdeb", "w_llm", "neg_bias")},
            "rule": "3-model stack replaces the 2-model stack iff train-OOF mean3 gain >= margin",
        }
        if args.bootstrap:
            choices[f"track{t}"]["dev_bootstrap_stack3_minus_stack2"] = bootstrap(
                dev, preds["stack3"], preds["stack2"], args.bootstrap, args.seed)
        predict_and_score(dev, preds["stack3"], "dev", t, args.out, "stack3")
    table = pd.DataFrame(report)
    table.to_csv(args.out / "summary.csv", index=False)
    (args.out / "choices.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({k: {x: v[x] for x in ("go", "oof_gain", "w_llm")} for k, v in choices.items()}, indent=1))


if __name__ == "__main__":
    main()

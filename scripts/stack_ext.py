"""Two extensions of the main-system stack, chosen on train OOF (pre-registered in PLAN.md 2d, 15/11).

  A  current: 2 models (Qwen for the track + mDeBERTa), grid over w and NEG bias
  B  + EQ bias (only EQ and NEG may get a bias: a FWD/BWD bias would break self_swap's
     reversal consistency, since Rev swaps FWD and BWD)
  C  3 models on tracks 1-3: mono Qwen + 4-track Qwen + mDeBERTa, weights on the simplex
  D  B and C together
Score = sum_i w_i S_i + neg_bias [l=NEG] + eq_bias [l=EQ], S_i = self_swap score of model i.
Rule: a variant replaces A only if its train-OOF mean3 objective is >= A + 0.003; D must also
beat B and C by 0.003. Dev (3-seed ensemble) is only reported.

    python scripts/stack_ext.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dico.data import LABEL2ID, LABELS, NEG, REV_PERM, add_reverse_index, load_split  # noqa: E402
from stack_qwen_mdeberta import dev_scores, oof_scores  # noqa: E402
from tune_neg_bias import local_metrics  # noqa: E402

EQ = LABELS.index("EQUIVALENCE")
NEG_ID = LABEL2ID[NEG]
MARGIN = 0.003
W_STEP = 0.05
NEG_GRID = np.round(np.arange(-20.0, 2.0 + 0.25, 0.5), 6)
EQ_GRID = np.round(np.arange(-3.0, 3.0 + 0.25, 0.5), 6)
REV = np.asarray(REV_PERM)


class FastMetrics:
    """Vectorised local_metrics (same definitions; checked against it below)."""

    def __init__(self, df: pd.DataFrame):
        self.gold = df.label.map(LABEL2ID).to_numpy()
        rev = df.rev_idx.to_numpy()
        i = np.flatnonzero(rev > np.arange(len(rev)))
        self.i, self.j = i, rev[i]
        self.support = np.bincount(self.gold, minlength=4)

    def __call__(self, pred: np.ndarray) -> dict[str, float]:
        g = self.gold
        tp = np.bincount(g[pred == g], minlength=4)
        npred = np.bincount(pred, minlength=4)
        prec = np.divide(tp, npred, out=np.zeros(4), where=npred > 0)
        rec = np.divide(tp, self.support, out=np.zeros(4), where=self.support > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
        pi, pj = pred[self.i], pred[self.j]
        return {"weighted_f1": float((f1 * self.support).sum() / self.support.sum()),
                "soft_cons": float(((pj == REV[pi]) & (pi != NEG_ID)).mean()),
                "hard_cons": float(((pi == g[self.i]) & (pj == g[self.j])).mean())}


def mean3(m: dict) -> float:
    return (m["weighted_f1"] + m["soft_cons"] + m["hard_cons"]) / 3


def simplex(k: int) -> list[tuple[float, ...]]:
    n = round(1 / W_STEP)
    return [tuple(c / n for c in combo) for combo in itertools.product(range(n + 1), repeat=k)
            if sum(combo) == n]


def search(scores: list[np.ndarray], fm: FastMetrics, eq_grid: np.ndarray) -> dict:
    best = {"objective": -1.0}
    for w in simplex(len(scores)):
        base = sum(wi * s for wi, s in zip(w, scores))
        for eq in eq_grid:
            for neg in NEG_GRID:
                s = base.copy()
                s[:, NEG_ID] += neg
                s[:, EQ] += eq
                m = fm(s.argmax(1))
                obj = mean3(m)
                if obj > best["objective"]:
                    best = {"objective": obj, "w": [float(x) for x in w], "neg_bias": float(neg),
                            "eq_bias": float(eq), **m}
    return best


def apply(scores: list[np.ndarray], pick: dict) -> np.ndarray:
    s = sum(wi * x for wi, x in zip(pick["w"], scores))
    s[:, NEG_ID] += pick["neg_bias"]
    s[:, EQ] += pick["eq_bias"]
    return s.argmax(1)


def main() -> None:
    system = yaml.safe_load((ROOT / "configs" / "main_system.yaml").read_text(encoding="utf-8"))
    mdeb = system["mdeberta"]
    out = ROOT / "runs" / "stack_ext"
    out.mkdir(parents=True, exist_ok=True)
    rows, picks = [], {}
    for t, spec in system["tracks"].items():
        t = int(t)
        train = add_reverse_index(load_split("train", [t]).reset_index(drop=True))
        dev = add_reverse_index(load_split("dev", [t]).reset_index(drop=True))
        fm = FastMetrics(train)
        names = [spec["qwen"], mdeb]
        if t != 4:
            names3 = [spec["qwen"], "qwen_cls_all", mdeb]
        oof = {n: oof_scores(n, train) for n in set(names) | (set(names3) if t != 4 else set())}
        dv = {n: dev_scores(n, t) for n in oof}
        # sanity: fast metrics == reference metrics
        p0 = (oof[names[0]] + oof[names[1]]).argmax(1)
        ref = local_metrics(train, [LABELS[k] for k in p0])
        assert all(abs(ref[k] - v) < 1e-9 for k, v in fm(p0).items()), (ref, fm(p0))

        variants = {"A": (names, np.array([0.0])), "B": (names, EQ_GRID)}
        if t != 4:
            variants |= {"C": (names3, np.array([0.0])), "D": (names3, EQ_GRID)}
        found = {}
        for v, (ns, eqg) in variants.items():
            found[v] = search([oof[n] for n in ns], fm, eqg) | {"models": ns}
            dm = local_metrics(dev, [LABELS[k] for k in apply([dv[n] for n in ns], found[v])])
            rows.append({"track": t, "variant": v, "models": "+".join(ns), "w": found[v]["w"],
                         "neg_bias": found[v]["neg_bias"], "eq_bias": found[v]["eq_bias"],
                         "oof_mean3": round(found[v]["objective"], 4),
                         "gain_vs_A": round(found[v]["objective"] - found["A"]["objective"], 4),
                         **{f"dev_{k}": round(x, 4) for k, x in dm.items()},
                         "dev_mean3": round(mean3(dm), 4)})
        # pre-registered choice
        a = found["A"]["objective"]
        ok = {v: f["objective"] for v, f in found.items() if v in "BC" and f["objective"] >= a + MARGIN}
        choice = max(ok, key=ok.get) if ok else "A"
        if "D" in found and found["D"]["objective"] >= max(a, found["B"]["objective"],
                                                           found["C"]["objective"]) + MARGIN:
            choice = "D"
        picks[t] = {"variant": choice, **found[choice]}
        print(f"track {t}: chosen {choice}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    (out / "choices.json").write_text(json.dumps(picks, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

"""One report for the unattended batch: every run group, every dev track, from saved probs.

For each run group (runs/<name>/seed*/dev_track*_{probs,swap_probs}.npy):
  - self_swap at neg_bias 0, mean +- std over seeds
  - self_swap at the train-OOF bias (runs/oof/<name>/neg_bias.json), if it exists
  - the seed ensemble (mean of probs) at both biases
  - EN/ES/EU disagreement on the same instances (bias 0)
Metrics are computed locally with tune_neg_bias.local_metrics, which matches the official
scorer exactly (checked on tracks 1 and 4). Nothing here is selected on dev.

    python scripts/overnight_report.py --out runs/overnight_report.md
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

from dico.data import add_reverse_index, load_split  # noqa: E402
from dico.decode import self_swap  # noqa: E402
from tune_neg_bias import local_metrics  # noqa: E402
from xling_agreement import key  # noqa: E402

GROUPS = [
    ("m0_mdeberta_xnli_pair_all", "4-label head, train 4 tracks, 3 epochs (current system)"),
    ("full_1ep", "4-label head, train 4 tracks, 1 epoch"),
    ("full_2ep", "4-label head, train 4 tracks, 2 epochs"),
    ("quick_base", "4-label head, train 3 mono tracks, 3 epochs"),
    ("bientail_mono", "bi-entailment head, train 3 mono tracks"),
    ("bientail_all", "bi-entailment head, train 4 tracks"),
    ("quick_js", "xling js, mono (method 1, 1 seed)"),
    ("quick_en", "xling en-teacher, mono (method 1, 1 seed)"),
    ("qwen_cls_en", "Qwen2.5-1.5B LoRA, 4-label head, train EN only (Kaggle)"),
    ("qwen_cls_es", "Qwen2.5-1.5B LoRA, 4-label head, train ES only (Kaggle)"),
    ("qwen_cls_eu", "Qwen2.5-1.5B LoRA, 4-label head, train EU only (Kaggle)"),
]
LANG = {1: "en", 2: "es", 3: "eu", 4: "mixed"}
METRICS = ("weighted_f1", "soft_cons", "hard_cons")


def load_probs(run: Path, track: int):
    seeds = []
    for seed_dir in sorted(run.glob("seed*")):
        p, s = seed_dir / f"dev_track{track}_probs.npy", seed_dir / f"dev_track{track}_swap_probs.npy"
        if p.exists() and s.exists():
            seeds.append((seed_dir.name, np.load(p), np.load(s)))
    return seeds


def oof_bias(name: str) -> float | None:
    path = ROOT / "runs" / "oof" / name / "neg_bias.json"
    return json.loads(path.read_text(encoding="utf-8"))["neg_bias"] if path.exists() else None


def fmt(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{np.mean(values):.3f}" + (f"±{np.std(values):.3f}" if len(values) > 1 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "overnight_report.md")
    args = ap.parse_args()
    dev = {t: add_reverse_index(load_split("dev", [t])) for t in LANG}

    rows, agree_rows = [], []
    for name, desc in GROUPS:
        run = ROOT / "runs" / name
        if not run.exists():
            continue
        tuned = oof_bias(name)
        seed_preds: dict[str, dict[int, list[str]]] = {}
        for t, df in dev.items():
            seeds = load_probs(run, t)
            if not seeds:
                continue
            for bias_name, bias in (("0", 0.0), ("oof", tuned)):
                if bias is None:
                    continue
                per_seed = []
                for seed, p, s in seeds:
                    labels = self_swap(df, p, s, bias)
                    per_seed.append(local_metrics(df, labels))
                    if bias_name == "0":
                        seed_preds.setdefault(seed, {})[t] = labels
                ens = local_metrics(df, self_swap(df, np.mean([x[1] for x in seeds], 0),
                                                  np.mean([x[2] for x in seeds], 0), bias))
                row = {"group": name, "track": LANG[t], "bias": f"{bias_name} ({bias:g})",
                       "seeds": len(seeds)}
                for m in METRICS:
                    row[m] = fmt([x[m] for x in per_seed])
                    row[f"ens_{m}"] = f"{ens[m]:.3f}"
                rows.append(row)
        for seed, preds in seed_preds.items():  # cross-language agreement, bias 0
            if not all(t in preds for t in (1, 2, 3)):
                continue
            series = {t: pd.Series(preds[t], index=key(dev[t].instance_id)) for t in (1, 2, 3)}
            idx = series[1].index
            same = (series[1] == series[2].loc[idx]) & (series[2].loc[idx] == series[3].loc[idx])
            agree_rows.append({"group": name, "seed": seed, "disagree": round(float(1 - same.mean()), 4)})

    table = pd.DataFrame(rows)
    lines = ["# Overnight report", "",
             "Dev scores are held-out: biases come from train OOF, nothing is chosen on dev.",
             "Cells: mean±std over seeds; `ens_*` = seed ensemble (mean probabilities).", "",
             "## Groups", ""]
    lines += [f"- `{n}`: {d}" for n, d in GROUPS if (ROOT / "runs" / n).exists()]
    for track in LANG.values():
        part = table[table.track == track] if len(table) else table
        if len(part):
            lines += ["", f"## Track {track}", "", part.drop(columns="track").to_markdown(index=False)]
    if agree_rows:
        agree = pd.DataFrame(agree_rows).groupby("group").disagree.agg(["mean", "std", "count"]).round(4)
        lines += ["", "## EN/ES/EU disagreement on the same instances (bias 0)", "", agree.to_markdown()]
    mt = ROOT / "runs" / "mt_views_1.3B" / "summary.csv"
    if mt.exists():
        lines += ["", "## MT views, NLLB-1.3B (last pre-registered attempt)", "",
                  pd.read_csv(mt).round(4).to_markdown(index=False)]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

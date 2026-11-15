"""One command from input CSVs to a CodaBench ZIP with the main system (configs/main_system.yaml).

Per track: probs of every (model, seed) on (A, B) and (B, A) -> mean over seeds -> self_swap
scores -> w * S_qwen + (1 - w) * S_mdeb + neg_bias * [NEG] -> labels. Every row is decoded from
its own text only (PLAN.md 2c).

Probabilities come from
  --probs model   the saved models (runs/<name>/seed<k>/model, needs keep_model: true; GPU), or
  --probs saved   the dev probs written at training time (dev only; CPU check of the pipeline).
With --probs model on dev, the probs are also compared with the training-time ones.

    # test phase, on Kaggle with the saved runs attached:
    python scripts/predict_final.py --split test --input_dir <test csv dir> --runs <runs dir>
    # pipeline check on dev:
    python scripts/predict_final.py --split dev --probs saved
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dico.data import LABEL2ID, LABELS, NEG, add_reverse_index, read_csv  # noqa: E402
from dico.decode import self_swap_scores  # noqa: E402
from dico.evaluate import write_predictions  # noqa: E402

DATA = ROOT / "external" / "SemEval-2027-Task-2-DiCo-NLI" / "final_data"


def input_file(input_dir: Path, split: str, track: int) -> Path:
    """The participant CSV of a track (texts, no gold needed); reference/template files are skipped."""
    hits = [p for p in sorted(input_dir.glob(f"*{split}_track{track}_*.csv"))
            if not p.name.endswith(("_reference.csv", "_submission_template.csv"))]
    if len(hits) != 1:
        raise SystemExit(f"expected one input CSV for {split} track {track} in {input_dir}, got {hits}")
    return hits[0]


def run_probs(mode: str, runs: Path, name: str, seeds: list[int], frames: dict[int, pd.DataFrame],
              split: str, cache: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Seed-mean (probs, swap_probs) of run `name` for each track in frames."""
    per_seed = []
    for seed in seeds:
        run = runs / name / f"seed{seed}"
        if mode == "saved":
            per_seed.append({t: (np.load(run / f"dev_track{t}_probs.npy"),
                                 np.load(run / f"dev_track{t}_swap_probs.npy")) for t in frames})
            continue
        from dico.predict import predict_run

        tracks = list(frames)
        got = dict(zip(tracks, predict_run(run, [frames[t] for t in tracks])))
        for t, (p, s) in got.items():
            np.save(cache / f"{name}_seed{seed}_{split}_track{t}_probs.npy", p)
            np.save(cache / f"{name}_seed{seed}_{split}_track{t}_swap_probs.npy", s)
            saved = run / f"{split}_track{t}_probs.npy"
            if saved.exists():  # dev: same model, same inputs -> should match training-time probs
                ref = np.load(saved)
                print(f"  check {name} seed{seed} track{t}: max |dp| = {np.abs(p - ref).max():.2e}, "
                      f"argmax agree = {(p.argmax(1) == ref.argmax(1)).mean():.4f}")
        per_seed.append(got)
    return {t: (np.mean([s[t][0] for s in per_seed], 0), np.mean([s[t][1] for s in per_seed], 0))
            for t in frames}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--input_dir", type=Path, help="default: official final_data/<split>")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--system", type=Path, default=ROOT / "configs" / "main_system.yaml")
    ap.add_argument("--probs", choices=["model", "saved"], default="model")
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", type=Path, help="default: submissions/<split>_main")
    args = ap.parse_args()
    if args.probs == "saved" and args.split != "dev":
        ap.error("--probs saved only exists for dev")
    input_dir = args.input_dir or DATA / args.split
    out = args.out or ROOT / "submissions" / f"{args.split}_main"
    cache = out / "probs"
    cache.mkdir(parents=True, exist_ok=True)
    system = yaml.safe_load(args.system.read_text(encoding="utf-8"))
    spec = {int(t): v for t, v in system["tracks"].items() if int(t) in args.tracks}

    frames = {t: read_csv(input_file(input_dir, args.split, t)) for t in spec}
    for t, df in frames.items():
        print(f"track{t}: {len(df)} rows from {input_file(input_dir, args.split, t).name}")
    needs: dict[str, dict[int, pd.DataFrame]] = {system["mdeberta"]: frames}
    for t, s in spec.items():
        needs.setdefault(s["qwen"], {})[t] = frames[t]
    probs = {name: run_probs(args.probs, args.runs, name, system["seeds"], fr, args.split, cache)
             for name, fr in needs.items()}

    paths = {}
    for t, s in spec.items():
        sq = self_swap_scores(*probs[s["qwen"]][t])
        sm = self_swap_scores(*probs[system["mdeberta"]][t])
        scores = s["w_qwen"] * sq + (1 - s["w_qwen"]) * sm
        scores[:, LABEL2ID[NEG]] += s["neg_bias"]
        labels = [LABELS[i] for i in scores.argmax(1)]
        paths[t] = write_predictions(frames[t], labels, out / f"track{t}_predictions.csv")
        if "label" in frames[t]:
            from tune_neg_bias import local_metrics

            m = local_metrics(add_reverse_index(frames[t]), labels)
            print(f"track{t} (local scorer): " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))

    cmd = [sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--split", args.split,
           "--template_dir", str(input_dir), "--out", str(out / f"{args.split}_main.zip")]
    for t, p in paths.items():
        cmd += [f"--track{t}", str(p)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

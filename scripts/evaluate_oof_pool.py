"""Fit the equivariant meta-pool on train OOF logits and evaluate untouched dev.

Requires outputs from ``scripts/generate_oof.py`` for mDeBERTa-all and
DeBERTa-English.  Produces official dev scores, leave-one-view-out ablations,
paired bootstrap confidence intervals, and frozen parameters for future test.
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

from dico.data import LABEL2ID, LABELS, REV_PERM, load_split  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402
from dico.meta_pool import MetaPoolModel, fit_meta_pool, predict_meta_pool, uniform_pool  # noqa: E402


VIEW_SPECS = [
    ("mdeberta_en", "mdeberta", 1),
    ("mdeberta_es", "mdeberta", 2),
    ("mdeberta_eu", "mdeberta", 3),
    ("mdeberta_mixed", "mdeberta", 4),
    ("deberta_en", "deberta", 1),
]


def aligned_group(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    logp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0))
    flipped = df.instance_id.str.endswith("__flipped").to_numpy()
    logp[flipped] = logp[flipped][:, REV_PERM]
    frame = pd.DataFrame(logp, columns=LABELS)
    frame["pair_id"] = df.pair_id.to_numpy()
    return frame.groupby("pair_id")[LABELS].mean()


def load_oof(name: str, tracks: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
    path = ROOT / "runs" / "oof" / name
    rows = pd.read_csv(path / "train_oof_rows.csv", dtype=str, keep_default_na=False)
    probs = np.load(path / "train_oof_probs.npy")
    df = load_split("train", tracks).reset_index(drop=True)
    if len(rows) != len(df) or not np.array_equal(rows.instance_id, df.instance_id):
        raise RuntimeError(f"OOF row manifest does not match current train data: {path}")
    return df, probs


def load_dev_probs(run_name: str, track: int, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_split("dev", [track]).reset_index(drop=True)
    path = ROOT / "runs" / run_name / f"seed{seed}" / f"dev_track{track}_probs.npy"
    return df, np.load(path)


def canonical_gold(df: pd.DataFrame) -> pd.Series:
    rows = []
    for row in df.itertuples(index=False):
        label = LABEL2ID[row.label]
        if row.instance_id.endswith("__flipped"):
            label = REV_PERM[label]
        rows.append((row.pair_id, label))
    grouped = pd.DataFrame(rows, columns=["pair_id", "label"]).groupby("pair_id").label
    if not grouped.nunique().eq(1).all():
        raise RuntimeError("canonical gold labels disagree within pair_id")
    return grouped.first().sort_index()


def collect() -> tuple[
    dict[int, pd.DataFrame], pd.Index, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    mtrain, mprob = load_oof("m0_mdeberta_base_all", [1, 2, 3, 4])
    etrain, eprob = load_oof("m0_deberta_base_en", [1])
    train_frames = {}
    dev_frames = {}
    dev_dfs = {track: load_split("dev", [track]).reset_index(drop=True) for track in range(1, 5)}
    for view, family, track in VIEW_SPECS:
        if family == "mdeberta":
            mask = mtrain.track.astype(int).to_numpy() == track
            train_frames[view] = aligned_group(mtrain.loc[mask].reset_index(drop=True), mprob[mask])
            ddf, dprob = load_dev_probs("m0_mdeberta_base_all", track)
        else:
            train_frames[view] = aligned_group(etrain, eprob)
            ddf, dprob = load_dev_probs("m0_deberta_base_en", track)
        dev_frames[view] = aligned_group(ddf, dprob)

    names = [spec[0] for spec in VIEW_SPECS]
    train_ids = train_frames[names[0]].index
    dev_ids = dev_frames[names[0]].index
    for name in names[1:]:
        train_ids = train_ids.intersection(train_frames[name].index)
        dev_ids = dev_ids.intersection(dev_frames[name].index)
    train_ids, dev_ids = train_ids.sort_values(), dev_ids.sort_values()
    x_train = np.stack([train_frames[n].loc[train_ids].to_numpy() for n in names], axis=1)
    x_dev = np.stack([dev_frames[n].loc[dev_ids].to_numpy() for n in names], axis=1)
    y_train = canonical_gold(load_split("train", [1])).loc[train_ids].to_numpy(dtype=int)
    y_dev = canonical_gold(dev_dfs[1]).loc[dev_ids].to_numpy(dtype=int)
    return dev_dfs, dev_ids, x_train, y_train, x_dev, y_dev


def expand(df: pd.DataFrame, pair_ids: pd.Index, source_pred: np.ndarray) -> list[str]:
    lookup = dict(zip(pair_ids, source_pred))
    labels = []
    for row in df.itertuples(index=False):
        label = int(lookup[row.pair_id])
        if row.instance_id.endswith("__flipped"):
            label = REV_PERM[label]
        labels.append(LABELS[label])
    return labels


def paired_stats(
    y: np.ndarray,
    full: np.ndarray,
    baseline: np.ndarray,
    sample_weight: np.ndarray,
    draws: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    def score(pred, idx) -> float:
        return float(f1_score(y[idx], pred[idx], average="weighted", sample_weight=sample_weight[idx]))

    all_idx = np.arange(len(y))
    observed = score(full, all_idx) - score(baseline, all_idx)
    deltas = np.empty(draws)
    permuted = np.empty(draws)
    for b in range(draws):
        idx = rng.integers(0, len(y), size=len(y))
        deltas[b] = score(full, idx) - score(baseline, idx)
        swap = rng.random(len(y)) < 0.5
        p1 = np.where(swap, baseline, full)
        p0 = np.where(swap, full, baseline)
        permuted[b] = score(p1, all_idx) - score(p0, all_idx)
    return {
        "observed_delta": observed,
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci95": list(map(float, np.quantile(deltas, [0.025, 0.975]))),
        "probability_delta_positive": float((deltas > 0).mean()),
        "paired_randomization_p": float((np.abs(permuted) >= abs(observed)).mean()),
        "draws": draws,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "oof_meta_pool")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev_dfs, dev_ids, x_train, y_train, x_dev, y_dev = collect()
    names = [spec[0] for spec in VIEW_SPECS]
    methods: dict[str, np.ndarray] = {"uniform": uniform_pool(x_dev)}
    train_methods: dict[str, np.ndarray] = {"uniform": uniform_pool(x_train)}
    models: dict[str, MetaPoolModel] = {}

    full = fit_meta_pool(x_train, y_train, names, l2=args.l2, calibrate=True)
    models["full"] = full
    methods["full"] = predict_meta_pool(x_dev, full)
    train_methods["full"] = predict_meta_pool(x_train, full)
    no_cal = fit_meta_pool(x_train, y_train, names, l2=args.l2, calibrate=False)
    models["no_calibration"] = no_cal
    methods["no_calibration"] = predict_meta_pool(x_dev, no_cal)
    train_methods["no_calibration"] = predict_meta_pool(x_train, no_cal)
    for omitted in names:
        keep = [i for i, name in enumerate(names) if name != omitted]
        model = fit_meta_pool(x_train[:, keep], y_train, [names[i] for i in keep], l2=args.l2)
        key = f"minus_{omitted}"
        models[key] = model
        methods[key] = predict_meta_pool(x_dev[:, keep], model)
        train_methods[key] = predict_meta_pool(x_train[:, keep], model)

    rows = []
    for method, source_pred in methods.items():
        for track, df in dev_dfs.items():
            row = predict_and_score(
                df, expand(df, dev_ids, source_pred), "dev", track, args.out / method, method
            )
            rows.append({"method": method, **row})
    results = pd.DataFrame(rows)
    macro = results.groupby("method")[["weighted_f1", "soft_cons", "hard_cons"]].mean()
    macro["oof_fit_weighted_f1"] = pd.Series(
        {
            method: f1_score(y_train, pred, average="weighted")
            for method, pred in train_methods.items()
        }
    )
    selected_method = str(macro.oof_fit_weighted_f1.idxmax())
    results.to_csv(args.out / "ablation_by_track.csv", index=False)
    macro.to_csv(args.out / "ablation_macro4.csv")

    counts = dev_dfs[1].groupby("pair_id").size().loc[dev_ids].to_numpy(dtype=float)
    full_stats = paired_stats(
        y_dev, methods["full"], methods["uniform"], counts, args.bootstrap, args.seed
    )
    selected_stats = paired_stats(
        y_dev,
        methods[selected_method],
        methods["uniform"],
        counts,
        args.bootstrap,
        args.seed + 1,
    )
    payload = {
        "fit_data": "train group-OOF predictions grouped by pair_id",
        "models": {name: model.to_dict() for name, model in models.items()},
        "selected_by_oof_fit_f1": selected_method,
        "full_vs_uniform": full_stats,
        "selected_vs_uniform": selected_stats,
    }
    (args.out / "frozen_meta_model.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("full weights:", ", ".join(f"{n}={w:.3f}" for n, w in zip(names, full.weights)))
    print("temperatures:", ", ".join(f"{n}={t:.3f}" for n, t in zip(names, full.temperatures)))
    print("\nmacro4 ablation:")
    print(macro.sort_values("weighted_f1", ascending=False).round(4).to_string())
    print("\npaired bootstrap full vs uniform:")
    print(json.dumps(full_stats, indent=2))
    print("\npaired bootstrap OOF-selected vs uniform:")
    print(json.dumps(selected_stats, indent=2))
    print(f"\nselected using train OOF only: {selected_method}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

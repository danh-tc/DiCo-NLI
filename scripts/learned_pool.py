"""Learn a small, regularized multi-view pooling model on saved dev probabilities.

The meta-model has one non-negative weight per model/view family plus one bias
per class.  Five-fold out-of-fold (OOF) predictions give a non-leaky estimate
on dev; a final fit on all dev source pairs is saved for later test inference.

    python scripts/learned_pool.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.data import LABEL2ID, LABELS, REV_PERM, load_split  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _aligned_frame(run: Path, tracks: list[int], seed_dirs: list[Path]) -> pd.DataFrame:
    """Return aligned log-probability rows from one model/view family."""
    parts = []
    for seed_dir in seed_dirs:
        for track in tracks:
            path = seed_dir / f"dev_track{track}_probs.npy"
            if not path.exists():
                continue
            df = load_split("dev", [track]).reset_index(drop=True)
            logp = np.log(np.clip(np.load(path), 1e-12, 1.0))
            flipped = df.instance_id.str.endswith("__flipped").to_numpy()
            logp[flipped] = logp[flipped][:, REV_PERM]
            part = pd.DataFrame(logp, columns=LABELS)
            part["pair_id"] = df.pair_id.to_numpy()
            parts.append(part)
    if not parts:
        raise FileNotFoundError(f"no dev probabilities found under {run}")
    return pd.concat(parts, ignore_index=True).groupby("pair_id")[LABELS].mean()


def collect_features() -> tuple[
    dict[int, pd.DataFrame], pd.DataFrame, np.ndarray, list[str], np.ndarray
]:
    """Build one [view_group, class] tensor per source pair."""
    mrun = ROOT / "runs" / "m0_mdeberta_base_all"
    erun = ROOT / "runs" / "m0_deberta_base_en"
    mseeds = sorted(mrun.glob("seed*"))
    eseeds = sorted(erun.glob("seed*"))
    groups = {
        "mdeberta_en": _aligned_frame(mrun, [1], mseeds),
        "mdeberta_es": _aligned_frame(mrun, [2], mseeds),
        "mdeberta_eu": _aligned_frame(mrun, [3], mseeds),
        "mdeberta_mixed": _aligned_frame(mrun, [4], mseeds),
        "deberta_en": _aligned_frame(erun, [1], eseeds),
    }
    names = list(groups)
    common = groups[names[0]].index
    for frame in groups.values():
        common = common.intersection(frame.index)
    common = common.sort_values()
    if common.empty:
        raise RuntimeError("view groups have no common pair_id values")

    features = np.stack([groups[name].loc[common].to_numpy() for name in names], axis=1)
    dfs = {track: load_split("dev", [track]).reset_index(drop=True) for track in range(1, 5)}

    # Gold labels aligned to the canonical `original` direction.
    gold_rows = []
    for row in dfs[1].itertuples(index=False):
        label = LABEL2ID[row.label]
        if row.instance_id.endswith("__flipped"):
            label = REV_PERM[label]
        gold_rows.append((row.pair_id, label))
    gold = pd.DataFrame(gold_rows, columns=["pair_id", "label"]).groupby("pair_id").label
    if not gold.nunique().eq(1).all():
        raise RuntimeError("gold labels disagree after orientation alignment")
    y = gold.first().loc[common].to_numpy(dtype=int)
    return dfs, pd.DataFrame({"pair_id": common}), features, names, y


def _unpack(params: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    weights = _softmax(params[:n_groups])
    bias = params[n_groups:]
    bias = bias - bias.mean()
    return weights, bias


def fit_meta(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    n_groups = x.shape[1]

    def objective(params: np.ndarray) -> float:
        weights, bias = _unpack(params, n_groups)
        scores = np.einsum("ngc,g->nc", x, weights) + bias
        prob = _softmax(scores)
        nll = -np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1.0)).mean()
        # Shrink biases and discourage a degenerate single-view solution.
        uniform = np.full(n_groups, 1.0 / n_groups)
        return float(nll + l2 * ((weights - uniform) ** 2).sum() + l2 * (bias**2).sum())

    result = minimize(objective, np.zeros(n_groups + len(LABELS)), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"meta-model optimization failed: {result.message}")
    return result.x


def predict_source(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    weights, bias = _unpack(params, x.shape[1])
    return (np.einsum("ngc,g->nc", x, weights) + bias).argmax(axis=1)


def expand_labels(df: pd.DataFrame, pair_ids: pd.Series, source_pred: np.ndarray) -> list[str]:
    lookup = dict(zip(pair_ids, source_pred))
    labels = []
    for row in df.itertuples(index=False):
        label = int(lookup[row.pair_id])
        if row.instance_id.endswith("__flipped"):
            label = REV_PERM[label]
        labels.append(LABELS[label])
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "learned_pool")
    args = ap.parse_args()

    dfs, pair_ids, x, names, y = collect_features()
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.full(len(y), -1, dtype=int)
    fold_weights = []
    for train_idx, valid_idx in splitter.split(x, y):
        params = fit_meta(x[train_idx], y[train_idx], args.l2)
        oof[valid_idx] = predict_source(x[valid_idx], params)
        fold_weights.append(_unpack(params, len(names))[0])
    if (oof < 0).any():
        raise RuntimeError("OOF predictions are incomplete")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for track, df in dfs.items():
        labels = expand_labels(df, pair_ids.pair_id, oof)
        row = predict_and_score(df, labels, "dev", track, args.out, "learned_pool_oof")
        rows.append(row)

    final_params = fit_meta(x, y, args.l2)
    weights, bias = _unpack(final_params, len(names))
    payload = {
        "view_groups": names,
        "weights": dict(zip(names, map(float, weights))),
        "class_bias": dict(zip(LABELS, map(float, bias))),
        "cv_mean_weights": dict(zip(names, map(float, np.mean(fold_weights, axis=0)))),
        "folds": args.folds,
        "l2": args.l2,
        "fit_split": "dev",
        "note": "Use these full-dev parameters only for future test inference; OOF metrics estimate dev performance.",
    }
    (args.out / "meta_params.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result = pd.DataFrame(rows)
    result.to_csv(args.out / "oof_results.csv", index=False)
    print("learned weights:", ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights)))
    print(result[["track", "weighted_f1", "soft_cons", "hard_cons"]].round(4).to_string(index=False))
    print("macro4:")
    print(result[["weighted_f1", "soft_cons", "hard_cons"]].mean().round(4).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Nested-OOF selection and dev evaluation for the dynamic expert router."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.dynamic_gate import fit_dynamic_gate, predict_dynamic_gate, routing_weights  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402
from dico.meta_pool import fit_meta_pool, predict_meta_pool, uniform_pool  # noqa: E402
from evaluate_oof_pool import VIEW_SPECS, collect, expand, paired_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--l2", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--anchor", type=float, nargs="+", default=[1e-2, 1e-1, 1.0])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "dynamic_gate")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev_dfs, dev_ids, x_train, y_train, x_dev, y_dev = collect()
    names = [spec[0] for spec in VIEW_SPECS]
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(x_train, y_train))
    grid_rows = []
    grid_predictions: dict[tuple[float, float], np.ndarray] = {}

    for l2, anchor in itertools.product(args.l2, args.anchor):
        oof_pred = np.empty(len(y_train), dtype=int)
        for fit_idx, valid_idx in splits:
            model = fit_dynamic_gate(
                x_train[fit_idx], y_train[fit_idx], names, l2=l2, anchor=anchor
            )
            oof_pred[valid_idx] = predict_dynamic_gate(x_train[valid_idx], model)
        score = float(f1_score(y_train, oof_pred, average="weighted"))
        grid_predictions[(l2, anchor)] = oof_pred
        grid_rows.append({"l2": l2, "anchor": anchor, "nested_oof_weighted_f1": score})
        print(f"l2={l2:g} anchor={anchor:g} nested_oof_f1={score:.6f}", flush=True)

    grid = pd.DataFrame(grid_rows).sort_values(
        ["nested_oof_weighted_f1", "anchor", "l2"], ascending=[False, False, False]
    )
    best = grid.iloc[0]
    best_l2, best_anchor = float(best.l2), float(best.anchor)
    gate = fit_dynamic_gate(
        x_train, y_train, names, l2=best_l2, anchor=best_anchor
    )
    static = fit_meta_pool(x_train, y_train, names, l2=0.1, calibrate=True)
    methods = {
        "uniform": uniform_pool(x_dev),
        "static_pool": predict_meta_pool(x_dev, static),
        "dynamic_gate": predict_dynamic_gate(x_dev, gate),
    }

    rows = []
    for method, source_pred in methods.items():
        for track, df in dev_dfs.items():
            row = predict_and_score(
                df, expand(df, dev_ids, source_pred), "dev", track, args.out / method, method
            )
            rows.append({"method": method, **row})
    results = pd.DataFrame(rows)
    macro = results.groupby("method")[["weighted_f1", "soft_cons", "hard_cons"]].mean()
    results.to_csv(args.out / "scores_by_track.csv", index=False)
    macro.to_csv(args.out / "scores_macro4.csv")
    grid.to_csv(args.out / "nested_oof_grid.csv", index=False)

    counts = dev_dfs[1].groupby("pair_id").size().loc[dev_ids].to_numpy(dtype=float)
    dynamic_vs_uniform = paired_stats(
        y_dev, methods["dynamic_gate"], methods["uniform"], counts, args.bootstrap, args.seed
    )
    dynamic_vs_static = paired_stats(
        y_dev,
        methods["dynamic_gate"],
        methods["static_pool"],
        counts,
        args.bootstrap,
        args.seed + 1,
    )
    weights = routing_weights(x_dev, gate)
    oracle = np.any(x_dev.argmax(axis=2) == y_dev[:, None], axis=1)
    report = {
        "selection": "hyperparameters selected by nested CV on train group-OOF only",
        "best_l2": best_l2,
        "best_anchor": best_anchor,
        "best_nested_oof_weighted_f1": float(best.nested_oof_weighted_f1),
        "model": gate.to_dict(),
        "mean_dev_routing_weights": dict(zip(names, map(float, weights.mean(axis=0)))),
        "mean_dev_routing_entropy": float(
            -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1).mean()
        ),
        "dev_oracle_coverage": float(oracle.mean()),
        "dynamic_vs_uniform": dynamic_vs_uniform,
        "dynamic_vs_static": dynamic_vs_static,
    }
    (args.out / "dynamic_gate_model.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nselected:", {"l2": best_l2, "anchor": best_anchor})
    print("\ndev macro4:")
    print(macro.round(4).to_string())
    print("\ndynamic vs uniform:")
    print(json.dumps(dynamic_vs_uniform, indent=2))
    print("\ndynamic vs static:")
    print(json.dumps(dynamic_vs_static, indent=2))
    print("\nmean routing weights:", report["mean_dev_routing_weights"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

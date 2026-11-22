"""Generate leakage-free train OOF probabilities grouped by source pair_id.

Each fold holds out every language/direction variant of its source pairs.  The
result can train a meta-decoder without touching dev labels.

Examples:
    python scripts/generate_oof.py --config configs/m0_mdeberta_base_all.yaml
    python scripts/generate_oof.py --config configs/m0_deberta_base_en.yaml
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico import env  # noqa: E402
from dico.data import LABEL2ID, REV_PERM, load_split  # noqa: E402
from dico.decode import swap_texts  # noqa: E402
from dico.train import (  # noqa: E402
    PairAwareCollator,
    PairDataset,
    build_model,
    load_config,
    softmax,
    trainer_setup,
    uses_pair_inputs,
    xling_setting,
)


def source_labels(df: pd.DataFrame) -> pd.Series:
    rows = []
    for row in df.itertuples(index=False):
        label = LABEL2ID[row.label]
        if row.instance_id.endswith("__flipped"):
            label = REV_PERM[label]
        rows.append((row.pair_id, label))
    grouped = pd.DataFrame(rows, columns=["pair_id", "label"]).groupby("pair_id").label
    if not grouped.nunique().eq(1).all():
        bad = grouped.nunique()[lambda x: x != 1].index[:5].tolist()
        raise RuntimeError(f"orientation-aligned gold disagreement for pair_ids: {bad}")
    return grouped.first().sort_index()


def training_args(
    cfg: dict, out: Path, seed: int, precision: str, pair_aware: bool = False
) -> TrainingArguments:
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        warmup = {"warmup_ratio": cfg["warmup_ratio"]}
    else:
        warmup = {"warmup_steps": float(cfg["warmup_ratio"])}
    return TrainingArguments(
        **warmup,
        output_dir=str(out),
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        weight_decay=cfg["weight_decay"],
        label_smoothing_factor=cfg.get("label_smoothing", 0.0),
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        disable_tqdm=False,
        remove_unused_columns=not pair_aware,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out = args.out or ROOT / "runs" / "oof" / cfg["name"]
    probs_path = out / "train_oof_probs.npy"
    if probs_path.exists() and not args.overwrite:
        raise SystemExit(f"{probs_path} exists; pass --overwrite to regenerate")
    out.mkdir(parents=True, exist_ok=True)

    env.set_seed(args.seed)
    precision = env.precision()
    df = load_split("train", cfg["train_tracks"]).reset_index(drop=True)
    sources = source_labels(df)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_of_pair: dict[str, int] = {}
    for fold, (_, valid) in enumerate(splitter.split(sources.index, sources.to_numpy())):
        fold_of_pair.update({sources.index[i]: fold for i in valid})
    row_fold = df.pair_id.map(fold_of_pair).to_numpy(dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Same objective as dico.train, so OOF probabilities match the final model's behaviour.
    pair_aware = uses_pair_inputs(cfg)
    xling = xling_setting(cfg, pair_aware)
    collator = PairAwareCollator(tokenizer) if pair_aware else DataCollatorWithPadding(tokenizer)
    template = cfg.get("template")
    oof = np.full((len(df), len(LABEL2ID)), np.nan, dtype=np.float32)
    oof_swap = np.full_like(oof, np.nan)
    fold_reports = []
    started = time.time()

    for fold in range(args.folds):
        fold_seed = args.seed + fold
        env.set_seed(fold_seed)
        train_idx = np.flatnonzero(row_fold != fold)
        valid_idx = np.flatnonzero(row_fold == fold)
        train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]
        if cfg.get("augment_transitive"):  # from this fold's train rows only, never the held-out pairs
            from dico.augment import augment, text_pairs

            train_df = augment(train_df, exclude=text_pairs(valid_df))
        print(f"\nfold={fold + 1}/{args.folds} train={len(train_df)} valid={len(valid_df)}")
        model = build_model(cfg, tokenizer, precision)
        trainer_cls, extra = trainer_setup(cfg, model)
        # Folds group whole pair_ids, so every cross-lingual partner stays in the same fold.
        train_ds = PairDataset(train_df, tokenizer, cfg["max_len"], template, pair_aware, xling)
        trainer = trainer_cls(
            model=model,
            args=training_args(cfg, out / f"fold{fold}", fold_seed, precision, pair_aware),
            train_dataset=train_ds,
            data_collator=collator,
            **extra,
        )
        t0 = time.time()
        trainer.train()
        for frame, target in ((valid_df, oof), (swap_texts(valid_df), oof_swap)):
            ds = PairDataset(frame, tokenizer, cfg["max_len"], template, pair_aware)
            pred = trainer.predict(ds).predictions
            target[valid_idx] = softmax(np.asarray(pred)).astype(np.float32)
        fold_reports.append({
            "fold": fold,
            "seed": fold_seed,
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "minutes": round((time.time() - t0) / 60, 3),
        })
        del trainer, model, pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not (np.isfinite(oof).all() and np.isfinite(oof_swap).all()):
        raise RuntimeError("OOF probability matrix contains missing/non-finite values")
    np.save(probs_path, oof)
    np.save(out / "train_oof_swap_probs.npy", oof_swap)
    pd.DataFrame({
        "instance_id": df.instance_id,
        "pair_id": df.pair_id,
        "track": df.track,
        "fold": row_fold,
    }).to_csv(out / "train_oof_rows.csv", index=False)
    report = {
        "config": cfg,
        "folds": args.folds,
        "seed": args.seed,
        "precision": precision,
        "total_minutes": round((time.time() - started) / 60, 3),
        "fold_reports": fold_reports,
    }
    (out / "oof_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {probs_path} shape={oof.shape} total={report['total_minutes']:.2f} min")


if __name__ == "__main__":
    main()

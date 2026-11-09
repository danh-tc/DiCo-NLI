"""Fine-tune a cross-encoder and score it on dev.

    python -m dico.train --config configs/m0_deberta_base_en.yaml --seed 1
    python -m dico.train --config configs/m0_deberta_base_en.yaml --seed 1 --set epochs=3 lr=3e-5

Output: runs/<name>/seed<seed>/ with config.yaml, dev probs, predictions,
official scores and results.json. Same command locally and on Kaggle.
"""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from . import env
from .data import LABEL2ID, LABELS, add_reverse_index, load_split
from .decode import DECODERS
from .evaluate import predict_and_score, save_json


def load_config(path: str, overrides: list[str]) -> dict:
    path = Path(path)
    cfg = yaml.safe_load((path.parent / "base.yaml").read_text(encoding="utf-8")) or {}
    cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    cfg.setdefault("name", path.stem)
    return cfg


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.enc = tokenizer(list(df.text1), list(df.text2), truncation=True, max_length=max_len)
        self.labels = df.label.map(LABEL2ID).tolist() if "label" in df else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {"weighted_f1": f1_score(labels, preds, average="weighted")}


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if cfg.get("method", "m0") != "m0":
        raise NotImplementedError(f"method {cfg['method']} not implemented yet")
    env.set_seed(args.seed)
    prec = env.precision()
    run_dir = env.runs_dir() / cfg["name"] / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": args.seed}), encoding="utf-8")
    print(env.describe())
    print(f"run_dir={run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    ).float()  # transformers 5 loads the checkpoint dtype (fp16 for deberta-v3) -> NaN with AdamW.
    # Master weights must be fp32; bf16/fp16 below is autocast (mixed precision) only.

    train_df = load_split("train", cfg["train_tracks"])
    dev_sets = {t: add_reverse_index(load_split("dev", [t])) for t in cfg["eval_tracks"]}
    train_ds = PairDataset(train_df, tokenizer, cfg["max_len"])
    dev_ds = {t: PairDataset(df, tokenizer, cfg["max_len"]) for t, df in dev_sets.items()}

    # transformers 5 dropped warmup_ratio (warmup_steps takes a float ratio); 4.x (Kaggle) still has it.
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        warmup = {"warmup_ratio": cfg["warmup_ratio"]}
    else:
        warmup = {"warmup_steps": float(cfg["warmup_ratio"])}
    targs = TrainingArguments(
        **warmup,
        output_dir=str(run_dir / "ckpt"),
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="no",  # fixed epochs, no best-on-dev selection: dev scores stay honest
        logging_steps=50,
        bf16=prec == "bf16",
        fp16=prec == "fp16",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds[cfg["eval_tracks"][0]],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    save_json(trainer.state.log_history, run_dir / "log_history.json")

    results = []
    for track, df in dev_sets.items():
        probs = softmax(trainer.predict(dev_ds[track]).predictions)
        np.save(run_dir / f"dev_track{track}_probs.npy", probs)
        for name in cfg["decoders"]:
            row = predict_and_score(df, DECODERS[name](df, probs), "dev", track, run_dir, name)
            results.append({"name": cfg["name"], "seed": args.seed, **row})

    save_json({"train_minutes": round(train_minutes, 2), "precision": prec, "results": results},
              run_dir / "results.json")
    if cfg.get("keep_model"):
        trainer.save_model(str(run_dir / "model"))
        tokenizer.save_pretrained(str(run_dir / "model"))
    print(pd.DataFrame(results).drop(columns=["name"]).round(4).to_string(index=False))
    print(f"train time: {train_minutes:.1f} min")


if __name__ == "__main__":
    main()

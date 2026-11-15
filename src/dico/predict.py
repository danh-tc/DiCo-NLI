"""Class probabilities from a saved run (needs keep_model: true), for any DiCo-NLI CSV.

A run dir is runs/<name>/seed<k>/ with config.yaml and model/. The input is encoded exactly as in
training (PairDataset with the run's template and max_len); only the (A, B) view is fed, so no
labels are needed (test files are unlabeled). Encoders are saved whole; LoRA runs save the adapter
plus the `score` head, and the base model is loaded from config.yaml's model_name.

    python -m dico.predict --run runs/qwen_cls_en/seed1 --input test_track1.csv --out probs/
    -> probs/<input stem>_probs.npy and <input stem>_swap_probs.npy (p(.|B,A), for self_swap)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from . import env
from .data import LABEL2ID, LABELS, read_csv
from .decode import swap_texts
from .train import PairDataset, softmax


def load_run(run_dir: Path, prec: str):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model_dir = run_dir / "model"
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if cfg.get("lora"):
        from peft import PeftModel

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec, torch.float32)
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], dtype=dtype, num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, model_dir)
        for name, p in model.named_parameters():  # as in training: LoRA + score head in fp32
            if "lora_" in name or "modules_to_save" in name:
                p.data = p.data.float()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).float()
    return cfg, tok, model.eval()


@torch.no_grad()
def predict_run(run_dir: Path, dfs: list, batch_size: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(probs, swap_probs)] of one saved run on each df (columns text1, text2); loads the model once."""
    prec = env.precision()
    cfg, tok, model = load_run(Path(run_dir), prec)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp, per_device_eval_batch_size=batch_size or cfg.get("eval_batch_size", 64),
            bf16=prec == "bf16", fp16=prec == "fp16", report_to="none", dataloader_num_workers=0)
        trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tok))
        for df in dfs:
            out = []
            for view in (df, swap_texts(df)):
                ds = PairDataset(view.drop(columns=["label"], errors="ignore"), tok, cfg["max_len"],
                                 cfg.get("template"), pair_aware=False)
                out.append(softmax(trainer.predict(ds).predictions))
            results.append((out[0], out[1]))
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="runs/<name>/seed<k> with model/")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    [(probs, swap)] = predict_run(args.run, [read_csv(args.input)])
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"{args.input.stem}_probs.npy", probs)
    np.save(args.out / f"{args.input.stem}_swap_probs.npy", swap)
    print(f"wrote probs for {len(probs)} rows to {args.out}")


if __name__ == "__main__":
    main()

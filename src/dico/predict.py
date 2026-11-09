"""Predict any DiCo-NLI CSV (e.g. the test set) with a saved model (needs keep_model: true).

    python -m dico.predict --model_dir runs/<name>/seed1/model --input test.csv --output preds.csv --decoder joint
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import add_reverse_index, read_csv
from .decode import DECODERS
from .evaluate import write_predictions


@torch.no_grad()
def predict_probs(model_dirs: list[str], df, max_len: int = 64, batch_size: int = 128) -> np.ndarray:
    """Average softmax probabilities over one or more models (simple ensemble)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = None
    for model_dir in model_dirs:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(device).eval()
        collate = DataCollatorWithPadding(tok)
        enc = tok(list(df.text1), list(df.text2), truncation=True, max_length=max_len)
        out = []
        for s in range(0, len(df), batch_size):
            batch = collate([{k: v[i] for k, v in enc.items()} for i in range(s, min(s + batch_size, len(df)))])
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        probs = np.concatenate(out)
        total = probs if total is None else total + probs
    return total / len(model_dirs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", nargs="+", required=True, help="one or more dirs (ensemble)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--decoder", default="joint", choices=list(DECODERS))
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    df = add_reverse_index(read_csv(args.input))
    probs = predict_probs(args.model_dir, df, args.max_len)
    write_predictions(df, DECODERS[args.decoder](df, probs), Path(args.output))
    print(f"wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()

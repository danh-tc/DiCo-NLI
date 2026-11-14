"""Predict any DiCo-NLI CSV (e.g. the test set) with a saved model (needs keep_model: true).

    python -m dico.predict --model_dir runs/<name>/seed1/model --input test.csv --output preds.csv

Default decoder is self_swap: each row is decoded from its own text only (PLAN.md 2c).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import add_reverse_index, read_csv
from .decode import DECODERS, NEEDS_SWAP, SAFE_DECODERS, decode, swap_texts
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

        def logits_of(a, b):
            enc = tok(list(a), list(b), truncation=True, max_length=max_len)
            out = []
            for s in range(0, len(df), batch_size):
                idx = range(s, min(s + batch_size, len(df)))
                batch = collate([{k: v[i] for k, v in enc.items()} for i in idx])
                with torch.autocast(device, enabled=device == "cuda"):
                    out.append(model(**{k: v.to(device) for k, v in batch.items()}).logits.float())
            return torch.cat(out)

        if getattr(model.config, "dico_method", "m0") == "bientail":
            # e(A=>B) and e(B=>A) from the NLI head, combined into the 4 labels.
            from .train import bientail_scores

            scores = bientail_scores(logits_of(df.text1, df.text2), logits_of(df.text2, df.text1),
                                     model.config.dico_entail_index)
        else:
            scores = logits_of(df.text1, df.text2)
        probs = torch.softmax(scores, -1).cpu().numpy()
        total = probs if total is None else total + probs
    return total / len(model_dirs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", nargs="+", required=True, help="one or more dirs (ensemble)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--decoder", default="self_swap", choices=list(DECODERS))
    ap.add_argument("--allow_unsafe", action="store_true",
                    help="allow decoders that look up other rows of the input (analysis only)")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--neg_bias", type=float, default=0.0, help="tuned on train OOF (tune_neg_bias.py)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--graph_lambda", type=float, default=1.0)
    ap.add_argument("--graph_time_limit", type=float)
    args = ap.parse_args()
    if args.decoder not in SAFE_DECODERS and not args.allow_unsafe:
        ap.error(f"{args.decoder} looks up other instances; use one of {SAFE_DECODERS} "
                 "or pass --allow_unsafe (not for submissions, see PLAN.md 2c)")

    df = read_csv(args.input)
    if args.decoder not in SAFE_DECODERS:
        df = add_reverse_index(df)
    probs = predict_probs(args.model_dir, df, args.max_len)
    probs_swap = None
    if args.decoder in NEEDS_SWAP:
        probs_swap = predict_probs(args.model_dir, swap_texts(df), args.max_len)
    config = {
        "neg_bias": args.neg_bias,
        "temperature": args.temperature,
        "graph_lambda": args.graph_lambda,
        "graph_time_limit": args.graph_time_limit,
    }
    write_predictions(df, decode(args.decoder, df, probs, config, probs_swap), Path(args.output))
    print(f"wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()

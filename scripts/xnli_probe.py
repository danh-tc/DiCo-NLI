"""Zero-shot probe: how far does the pretrained XNLI head get, asked in both directions?

No fine-tuning. For each row we take the XNLI 3-way log-probs (entailment, neutral,
contradiction) for (A, B) and for (B, A): 6 features. A small logistic regression is fit
on train features and decoded with self_swap (features of the swapped input are the
same 6 numbers in the other order), then scored on dev with the official scorer.

Also reports the pure rule from the bidirectional-entailment table
(both -> EQ, A=>B only -> FWD, B=>A only -> BWD, neither -> NEG) with threshold 0.5.

    python scripts/xnli_probe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.data import BWD, EQ, FWD, LABEL2ID, LABELS, NEG, load_split  # noqa: E402
from dico.decode import self_swap  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402

MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
ENT = 0  # id2label: 0 entailment, 1 neutral, 2 contradiction


@torch.no_grad()
def xnli_logp(model, tok, a: list[str], b: list[str], device: str, bs: int = 256) -> np.ndarray:
    collate = DataCollatorWithPadding(tok)
    enc = tok(a, b, truncation=True, max_length=64)
    out = []
    for s in range(0, len(a), bs):
        batch = collate([{k: v[i] for k, v in enc.items()} for i in range(s, min(s + bs, len(a)))])
        with torch.autocast(device, enabled=device == "cuda"):
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        out.append(torch.log_softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def features(model, tok, df: pd.DataFrame, device: str) -> tuple[np.ndarray, np.ndarray]:
    """[f(A,B), f(B,A)] and its swapped version [f(B,A), f(A,B)]."""
    fwd = xnli_logp(model, tok, list(df.text1), list(df.text2), device)
    bwd = xnli_logp(model, tok, list(df.text2), list(df.text1), device)
    return np.hstack([fwd, bwd]), np.hstack([bwd, fwd])


def table_rule(x: np.ndarray) -> list[str]:
    ab, ba = np.exp(x[:, ENT]) > 0.5, np.exp(x[:, 3 + ENT]) > 0.5
    return [EQ if p and q else FWD if p else BWD if q else NEG for p, q in zip(ab, ba)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "xnli_probe")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).float().to(device).eval()

    rows = []
    for t in args.tracks:
        train = load_split("train", [t])
        dev = load_split("dev", [t]).reset_index(drop=True)
        x_tr, _ = features(model, tok, train, device)
        x_dev, x_dev_swap = features(model, tok, dev, device)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_tr, train.label.map(LABEL2ID))
        order = np.argsort(clf.classes_)  # columns aligned with LABELS order
        p, p_swap = clf.predict_proba(x_dev)[:, order], clf.predict_proba(x_dev_swap)[:, order]
        systems = {
            "table_rule_0.5": table_rule(x_dev),
            "lr_argmax": [LABELS[i] for i in p.argmax(1)],
            "lr_self_swap": self_swap(dev, p, p_swap),
        }
        for name, labels in systems.items():
            r = predict_and_score(dev, labels, "dev", t, args.out, name)
            rows.append({"track": t, "system": name,
                         **{k: r[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Zero-shot probe for idea 2: does a shared sentence frame help phrase-level NLI?

Same setup as xnli_probe.py (pretrained mDeBERTa-XNLI, 6 bidirectional log-probs, LR fit on
train, self_swap on dev), but each phrase can be wrapped in a fixed frame first. Frames are
fixed a priori (PLAN.md, idea 2) and only use the instance's own text, so they are test-safe.

Systems per frame F:  orig (6 features), F (6 features on framed text), orig+F (12 features).
Paired bootstrap by pair_id for F and orig+F vs orig.

    python scripts/context_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dico.data import LABEL2ID, load_split  # noqa: E402
from dico.decode import self_swap  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402
from mt_views_eval import paired_bootstrap  # noqa: E402
from xnli_probe import MODEL, features  # noqa: E402

TRACK_LANG = {1: "en", 2: "es", 3: "eu"}
FRAMES = {
    "quote": {"en": 'The text says: "{x}".', "es": 'El texto dice: "{x}".', "eu": 'Testuak dio: "{x}".'},
    "ellipsis": {lang: "… {x} …" for lang in ("en", "es", "eu")},
}


def framed(df: pd.DataFrame, frame: str, lang: str) -> pd.DataFrame:
    tpl = FRAMES[frame][lang]
    return df.assign(text1=[tpl.format(x=x) for x in df.text1], text2=[tpl.format(x=x) for x in df.text2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "context_probe")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).float().to(device).eval()

    rows, cis = [], []
    for t in args.tracks:
        lang = TRACK_LANG[t]
        train = load_split("train", [t]).reset_index(drop=True)
        dev = load_split("dev", [t]).reset_index(drop=True)
        y = train.label.map(LABEL2ID)
        feats = {"orig": (features(model, tok, train, device)[0], *features(model, tok, dev, device))}
        for f in FRAMES:
            feats[f] = (features(model, tok, framed(train, f, lang), device)[0],
                        *features(model, tok, framed(dev, f, lang), device))
        systems = {"orig": feats["orig"]}
        for f in FRAMES:
            systems[f] = feats[f]
            systems[f"orig+{f}"] = tuple(np.hstack([a, b]) for a, b in zip(feats["orig"], feats[f]))
        preds = {}
        for name, (x_tr, x_dev, x_dev_swap) in systems.items():
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(x_tr, y)
            order = np.argsort(clf.classes_)
            p, p_swap = clf.predict_proba(x_dev)[:, order], clf.predict_proba(x_dev_swap)[:, order]
            preds[name] = self_swap(dev, p, p_swap)
            r = predict_and_score(dev, preds[name], "dev", t, args.out, name)
            rows.append({"track": t, "system": name,
                         **{k: r[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}})
        for name in preds:
            if name != "orig":
                cis.append({"track": t, "system": name, **paired_bootstrap(dev, preds["orig"], preds[name])})
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "summary.csv", index=False)
    (args.out / "bootstrap.json").write_text(json.dumps(cis, indent=2))
    print(table.round(4).to_string(index=False))
    print(pd.DataFrame(cis).to_string(index=False))


if __name__ == "__main__":
    main()

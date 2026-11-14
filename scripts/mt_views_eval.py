"""Do self-generated translation views help the NLI model? (go / no-go for N2)

For each monolingual dev track (language L) every row gets these views of the SAME instance:
  orig        the row itself
  mt:<X>      both phrases machine-translated L -> X with NLLB (legal at test time)
  mix:<X>     code-mixed: A kept in L, B translated to X, and the mirror (legal)
  gold:<X>    the organizers' verified translation of the same instance from track X
              (ANALYSIS ONLY: reads other tracks, i.e. an upper bound for MT views)

Each view is decoded with self_swap; pools average the self_swap scores over views with
equal weights (nothing is tuned on dev) and use the config's train-OOF neg_bias.

    python scripts/mt_views_eval.py --model_dir runs/xnli_pair_keep/seed1/model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.data import LABELS, load_split  # noqa: E402
from dico.decode import self_swap_scores, swap_texts  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402
from dico.mt import Translator, cached_translate  # noqa: E402

LANG_TRACK = {"en": 1, "es": 2, "eu": 3}


def key(ids: pd.Series) -> pd.Series:
    return ids.str.replace(r"__[a-z]{2}-[a-z]{2}__", "__", regex=True)


class NLI:
    def __init__(self, model_dir: str, max_len: int = 64, batch_size: int = 128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(self.device).eval()
        )
        self.collate = DataCollatorWithPadding(self.tok)
        self.max_len, self.batch_size = max_len, batch_size

    @torch.no_grad()
    def probs(self, a: list[str], b: list[str]) -> np.ndarray:
        enc = self.tok(a, b, truncation=True, max_length=self.max_len)
        out = []
        for s in range(0, len(a), self.batch_size):
            idx = range(s, min(s + self.batch_size, len(a)))
            batch = self.collate([{k: v[i] for k, v in enc.items()} for i in idx])
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                logits = self.model(**{k: v.to(self.device) for k, v in batch.items()}).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    def scores(self, a: list[str], b: list[str], neg_bias: float) -> np.ndarray:
        """self_swap scores; neg_bias is added once per view, so the mean over views keeps it."""
        frame = pd.DataFrame({"text1": a, "text2": b})
        swap = swap_texts(frame)
        return self_swap_scores(self.probs(a, b), self.probs(list(swap.text1), list(swap.text2)), neg_bias)


def build_views(df: pd.DataFrame, lang: str, cache: Path, translator: Translator | None) -> dict:
    views = {"orig": (list(df.text1), list(df.text2))}
    phrases = sorted(set(df.text1) | set(df.text2))
    for other in LANG_TRACK:
        if other == lang:
            continue
        mt = cached_translate(phrases, lang, other, cache, translator)
        a_mt, b_mt = [mt[x] for x in df.text1], [mt[x] for x in df.text2]
        views[f"mt:{other}"] = (a_mt, b_mt)
        views[f"mix:{other}"] = (list(df.text1) + a_mt, b_mt + list(df.text2))  # both mixings
        gold = load_split("dev", [LANG_TRACK[other]]).assign(k=lambda g: key(g.instance_id))
        gold = gold.set_index("k").loc[key(df.instance_id)]
        views[f"gold:{other}"] = (list(gold.text1), list(gold.text2))
    return views


def paired_bootstrap(df: pd.DataFrame, pred_a, pred_b, n: int = 2000, seed: int = 0) -> dict:
    """95% CI of (b - a) for weighted F1 and HardCons, resampling whole pair_ids so both
    directions of a pair stay together. Metrics match the official scorer (tune_neg_bias)."""
    from sklearn.metrics import f1_score

    from dico.data import add_reverse_index

    df = add_reverse_index(df)
    gold = df.label.to_numpy()
    a, b = np.asarray(pred_a), np.asarray(pred_b)
    groups = df.groupby("pair_id").indices
    names = list(groups)
    rev = df.rev_idx.to_numpy()
    hard_a = {g: all(a[i] == gold[i] for i in idx) for g, idx in groups.items()}
    hard_b = {g: all(b[i] == gold[i] for i in idx) for g, idx in groups.items()}
    paired = {g: len(idx) == 2 and rev[idx[0]] == idx[1] for g, idx in groups.items()}
    rng = np.random.default_rng(seed)
    d_f1, d_hard = [], []
    for _ in range(n):
        pick = rng.choice(len(names), size=len(names), replace=True)
        rows = np.concatenate([groups[names[k]] for k in pick])
        d_f1.append(f1_score(gold[rows], b[rows], average="weighted")
                    - f1_score(gold[rows], a[rows], average="weighted"))
        pg = [names[k] for k in pick if paired[names[k]]]
        d_hard.append(np.mean([hard_b[g] for g in pg]) - np.mean([hard_a[g] for g in pg]))
    ci = lambda x: [round(float(np.percentile(x, q)), 4) for q in (2.5, 97.5)]  # noqa: E731
    return {"d_f1_ci": ci(d_f1), "d_hard_ci": ci(d_hard)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--neg_bias", type=float, default=-10.0, help="train-OOF choice (neg_bias.json)")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "eu"])
    ap.add_argument("--mt_model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "mt_views")
    args = ap.parse_args()
    cache = args.out / "mt_cache"

    # Translate first (NLLB), free it, then load the NLI model.
    dfs = {lang: load_split("dev", [LANG_TRACK[lang]]).reset_index(drop=True) for lang in args.langs}
    translator = Translator(args.mt_model)
    views = {lang: build_views(df, lang, cache, translator) for lang, df in dfs.items()}
    del translator
    torch.cuda.empty_cache()

    nli = NLI(args.model_dir)
    rows = []
    for lang, df in dfs.items():
        n = len(df)
        scores = {}
        for name, (a, b) in views[lang].items():
            s = nli.scores(a, b, args.neg_bias)
            # mix views hold two mixings stacked: average them into one view
            scores[name] = (s[:n] + s[n:]) / 2 if name.startswith("mix:") else s
        others = [x for x in LANG_TRACK if x != lang]
        systems = {name: [name] for name in scores}
        systems["pool:orig+mt"] = ["orig"] + [f"mt:{x}" for x in others]
        systems["pool:orig+mt+mix"] = systems["pool:orig+mt"] + [f"mix:{x}" for x in others]
        systems["pool:orig+gold (analysis)"] = ["orig"] + [f"gold:{x}" for x in others]
        track = LANG_TRACK[lang]
        preds = {}
        for system, members in systems.items():
            pooled = np.mean([scores[m] for m in members], axis=0)
            labels = [LABELS[i] for i in pooled.argmax(axis=1)]
            preds[system] = labels
            tag = system.replace(":", "-").replace("+", "_").replace(" ", "").replace("(analysis)", "")
            row = predict_and_score(df, labels, "dev", track, args.out / lang, tag)
            safe = not any(m.startswith("gold:") for m in members)
            rows.append({"lang": lang, "system": system, "test_legal": safe,
                         **{k: row[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}})
        # Pre-registered comparisons against the single-view baseline.
        for system in ("pool:orig+mt", "pool:orig+mt+mix", "pool:orig+gold (analysis)"):
            ci = paired_bootstrap(df, preds["orig"], preds[system])
            next(r for r in rows if r["lang"] == lang and r["system"] == system).update(ci)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

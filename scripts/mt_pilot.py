"""MT pilot: are self-generated translations of short phrases good enough to serve as views?

Translates the English dev phrases (track 1) to Spanish and Basque with NLLB and compares
them with the organizers' verified translations (tracks 2/3, same pair_id and orientation).
The official translations are used only as a quality reference here; the planned decoder
translates each test instance itself and never reads other tracks (PLAN.md 2c).

Reports per target language:
  chrF            corpus chrF++ of MT vs. official phrase
  exact           share of phrases identical to the official translation (case-insensitive)
  collapse_mt     share of pairs with A != B in English but MT(A) == MT(B): the view loses the
                  distinction the label depends on (e.g. law / laws)
  collapse_gold   same rate for the official translations (the organizers filter these)

    python scripts/mt_pilot.py --model facebook/nllb-200-distilled-600M
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dico.data import load_split  # noqa: E402
from dico.mt import postprocess  # noqa: E402

NLLB = {"en": "eng_Latn", "es": "spa_Latn", "eu": "eus_Latn"}
TRACK = {"es": 2, "eu": 3}


def originals(track: int) -> pd.DataFrame:
    df = load_split("dev", [track])
    df = df[df.instance_id.str.endswith("__original")]
    return df.set_index("pair_id")[["text1", "text2", "label"]]


@torch.no_grad()
def translate(texts: list[str], tok, model, tgt: str, batch_size: int, device: str) -> list[str]:
    out = []
    for s in range(0, len(texts), batch_size):
        batch = tok(texts[s : s + batch_size], return_tensors="pt", padding=True).to(device)
        gen = model.generate(
            **batch,
            forced_bos_token_id=tok.convert_tokens_to_ids(NLLB[tgt]),
            num_beams=4,
            max_new_tokens=48,
        )
        hyps = tok.batch_decode(gen, skip_special_tokens=True)
        out.extend(postprocess(a, h) for a, h in zip(texts[s : s + batch_size], hyps))
    return out


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "mt_pilot")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, src_lang=NLLB["en"])
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    en = originals(1)
    phrases = sorted(set(en.text1) | set(en.text2))
    summary, rows = [], []
    for lang, track in TRACK.items():
        t0 = time.time()
        mt = dict(zip(phrases, translate(phrases, tok, model, lang, args.batch_size, device)))
        gold = originals(track).reindex(en.index)
        hyp, ref = [], []
        collapse_mt = collapse_gold = distinct = 0
        for pid, row in en.iterrows():
            g = gold.loc[pid]
            a_mt, b_mt = mt[row.text1], mt[row.text2]
            for h, r in ((a_mt, g.text1), (b_mt, g.text2)):
                hyp.append(h)
                ref.append(r)
            if norm(row.text1) != norm(row.text2):
                distinct += 1
                collapse_mt += norm(a_mt) == norm(b_mt)
                collapse_gold += norm(g.text1) == norm(g.text2)
            rows.append({"pair_id": pid, "lang": lang, "label": row.label,
                         "en_a": row.text1, "en_b": row.text2,
                         "mt_a": a_mt, "mt_b": b_mt, "gold_a": g.text1, "gold_b": g.text2})
        summary.append({
            "lang": lang,
            "chrF": round(CHRF(word_order=2).corpus_score(hyp, [ref]).score, 2),
            "exact": round(sum(norm(h) == norm(r) for h, r in zip(hyp, ref)) / len(hyp), 4),
            "collapse_mt": round(collapse_mt / distinct, 4),
            "collapse_gold": round(collapse_gold / distinct, 4),
            "pairs": len(en),
            "seconds": round(time.time() - t0, 1),
        })
    pd.DataFrame(rows).to_csv(args.out / "dev_track1_mt_views.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(args.out / "summary.csv", index=False)
    print(f"model={args.model} device={device}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

"""Idea B (PLAN.md 2d): a large instruction LLM as a zero-shot third expert for the stack.

For every row the LLM reads the label definitions and 4 hand-written examples (not from
train/dev), then we read its next-token log-probs of the four answer letters for (A, B) and
for (B, A). Nothing is trained, so the train-split probabilities are as clean as OOF ones and
the stack weights can be chosen on train exactly as for the other experts.

Outputs use the layout of a trained model, so scripts/stack_llm3.py (and the 2-model stack
helpers) read them unchanged:
    runs/oof/<name>/train_oof_{probs,swap_probs}.npy + train_oof_rows.csv   (train, tracks)
    runs/<name>/seed1/dev_track<t>_{probs,swap_probs}.npy                    (dev)

    python scripts/llm_expert.py --model Qwen/Qwen2.5-7B-Instruct --load_4bit --tracks 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico import env  # noqa: E402
from dico.data import BWD, EQ, FWD, LABEL2ID, LABELS, NEG, load_split  # noqa: E402
from dico.decode import self_swap  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402

LETTERS = {"A": EQ, "B": FWD, "C": BWD, "D": NEG}
SYSTEM = (
    "You judge the meaning relation between two short phrases taken from image captions or news "
    "headlines. The phrases may be in English, Spanish or Basque. Answer with one letter:\n"
    "A = equivalent: each phrase implies the other (same meaning).\n"
    "B = Phrase 1 implies Phrase 2, but not the other way (Phrase 1 is more specific).\n"
    "C = Phrase 2 implies Phrase 1, but not the other way (Phrase 2 is more specific).\n"
    "D = neither implies the other (different, unrelated or incompatible details)."
)
# hand-written, not taken from train/dev
SHOTS = [
    ("a large crowd", "a big crowd", "A"),
    ("an old red bicycle", "a bicycle", "B"),
    ("the minister", "the finance minister", "C"),
    ("three boys", "a young girl", "D"),
]


def prompt(tok, a: str, b: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}]
    for x, y, ans in SHOTS:
        msgs += [{"role": "user", "content": f"Phrase 1: {x}\nPhrase 2: {y}"},
                 {"role": "assistant", "content": ans}]
    msgs.append({"role": "user", "content": f"Phrase 1: {a}\nPhrase 2: {b}"})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_pairs(model, tok, pairs: list[tuple[str, str]], bs: int) -> dict:
    """{(a, b): probs over LABELS} from the next-token distribution over the 4 letters."""
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in LETTERS]
    cols = [LABEL2ID[LETTERS[x]] for x in LETTERS]
    texts = [prompt(tok, a, b) for a, b in pairs]
    order = np.argsort([len(t) for t in texts])  # length-sorted batches: less padding
    out = {}
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()  # left padding: last position is the answer
        p4 = torch.softmax(logits[:, letter_ids], -1).cpu().numpy()
        for row, i in enumerate(idx):
            probs = np.zeros(len(LABELS), dtype=np.float32)
            probs[cols] = p4[row]
            out[pairs[i]] = probs
        if (s // bs) % 20 == 0:
            print(f"  scored {min(s + bs, len(order))}/{len(order)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--name", default="llm7b_expert")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--tracks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows per split")
    args = ap.parse_args()

    runs = env.runs_dir()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    kw = {"dtype": dtype, "device_map": {"": 0}}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    t0 = time.time()
    data = {s: {t: load_split(s, [t]).reset_index(drop=True) for t in args.tracks} for s in ("train", "dev")}
    if args.limit:
        data = {s: {t: d.head(args.limit) for t, d in v.items()} for s, v in data.items()}
    pairs = sorted({p for v in data.values() for d in v.values()
                    for a, b in zip(d.text1, d.text2) for p in ((a, b), (b, a))})
    print(f"{len(pairs)} unique ordered pairs to score")
    probs = score_pairs(model, tok, pairs, args.bs)
    minutes = (time.time() - t0) / 60

    def lookup(d: pd.DataFrame):
        return (np.stack([probs[(a, b)] for a, b in zip(d.text1, d.text2)]),
                np.stack([probs[(b, a)] for a, b in zip(d.text1, d.text2)]))

    oof_dir = runs / "oof" / args.name
    oof_dir.mkdir(parents=True, exist_ok=True)
    train = pd.concat(data["train"].values(), ignore_index=True)
    p, q = lookup(train)
    np.save(oof_dir / "train_oof_probs.npy", p)
    np.save(oof_dir / "train_oof_swap_probs.npy", q)
    train[["instance_id", "pair_id", "track"]].assign(fold=-1).to_csv(oof_dir / "train_oof_rows.csv", index=False)

    seed_dir = runs / args.name / "seed1"
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, dev in data["dev"].items():
        p, q = lookup(dev)
        np.save(seed_dir / f"dev_track{t}_probs.npy", p)
        np.save(seed_dir / f"dev_track{t}_swap_probs.npy", q)
        for name, labels in (("argmax", [LABELS[i] for i in p.argmax(1)]), ("self_swap", self_swap(dev, p, q))):
            if args.limit:
                continue  # partial dev files cannot go through the official scorer
            r = predict_and_score(dev, labels, "dev", t, seed_dir, name)
            rows.append({"track": t, "decoder": name, **{k: r[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}})
    report = {"model": args.model, "load_4bit": args.load_4bit, "tracks": args.tracks,
              "pairs": len(pairs), "minutes": round(minutes, 2), "dev_bias0": rows}
    (seed_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rows:
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"scored {len(pairs)} pairs in {minutes:.1f} min")


if __name__ == "__main__":
    main()

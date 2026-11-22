"""Idea 2b: a meaningful, LLM-written shared sentence frame for each phrase pair (PLAN.md 2d).

For every pair the LLM writes ONE frame with a slot [X] that both phrases fit. The frame is
generated on the pair in a fixed order (min, max), so (A, B) and (B, A) share it and self_swap
stays reversal-consistent. Only the instance's own text is used (test-safe). Broken frames
(no [X]) fall back to the ellipsis frame.

Then the zero-shot pipeline of context_probe.py: mDeBERTa-XNLI, 6 bidirectional log-probs,
LR on train, self_swap on dev. Main comparison: orig+llm vs orig+ellipsis.

    python scripts/context_llm.py --track 1
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
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dico import env  # noqa: E402
from dico.data import LABEL2ID, load_split  # noqa: E402
from dico.decode import self_swap  # noqa: E402
from dico.evaluate import predict_and_score  # noqa: E402
from mt_views_eval import paired_bootstrap  # noqa: E402
from xnli_probe import MODEL, features  # noqa: E402

LLM = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = (
    "You write ONE short, natural English sentence that contains the slot [X]. "
    "Both given phrases must fit grammatically into [X]. The sentence should describe a "
    "realistic situation (a news headline or a photo caption), but it must not change, "
    "explain or add to the meaning of either phrase, and it must not say anything that makes "
    "one phrase true and the other false. Output only the sentence."
)
# hand-written, not taken from train/dev
SHOTS = [
    ("a black dog", "a dog", "In the photo, [X] is running across the park."),
    ("22 dead", "89 dead", "The earthquake left [X], local officials reported."),
    ("resigns", "steps down", "The finance minister [X] after the scandal."),
    ("on the beach", "near the water", "Two children are playing [X]."),
    ("new rules", "the law", "Parliament approved [X] on Tuesday."),
]
ELLIPSIS = "… {x} …"


def prompt(tok, a: str, b: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}]
    for x, y, f in SHOTS:
        msgs += [{"role": "user", "content": f"Phrase 1: {x}\nPhrase 2: {y}"},
                 {"role": "assistant", "content": f}]
    msgs.append({"role": "user", "content": f"Phrase 1: {a}\nPhrase 2: {b}"})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_frames(pairs: list[tuple[str, str]], cache: Path, llm: str = LLM, load_4bit: bool = False,
                    bs: int = 48) -> dict:
    frames = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [p for p in pairs if "|||".join(p) not in frames]
    if todo:
        tok = AutoTokenizer.from_pretrained(llm, padding_side="left")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # T4: fp16
        if load_4bit:
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            lm = AutoModelForCausalLM.from_pretrained(llm, quantization_config=q, device_map={"": 0}).eval()
        else:
            lm = AutoModelForCausalLM.from_pretrained(llm, dtype=dtype).cuda().eval()
        for s in range(0, len(todo), bs):
            chunk = todo[s:s + bs]
            enc = tok([prompt(tok, a, b) for a, b in chunk], return_tensors="pt", padding=True).to("cuda")
            out = lm.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
            texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for p, t in zip(chunk, texts):
                frames["|||".join(p)] = t.strip().split("\n")[0]
            print(f"  frames {min(s + bs, len(todo))}/{len(todo)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(frames, indent=1, ensure_ascii=False), encoding="utf-8")
        del lm
        torch.cuda.empty_cache()
    return frames


def canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def apply(df: pd.DataFrame, frames: dict | None) -> tuple[pd.DataFrame, float]:
    t1, t2, ok = [], [], 0
    for a, b in zip(df.text1, df.text2):
        f = frames.get("|||".join(canon(a, b)), "") if frames else ""
        if f.count("[X]") == 1:
            ok += 1
            t1.append(f.replace("[X]", a))
            t2.append(f.replace("[X]", b))
        else:
            t1.append(ELLIPSIS.format(x=a))
            t2.append(ELLIPSIS.format(x=b))
    return df.assign(text1=t1, text2=t2), ok / len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--out", type=Path, default=env.runs_dir() / "context_llm")
    ap.add_argument("--llm", default=LLM)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--gen_bs", type=int, default=48)
    args = ap.parse_args()
    t = args.track
    train = load_split("train", [t]).reset_index(drop=True)
    dev = load_split("dev", [t]).reset_index(drop=True)
    pairs = sorted({canon(a, b) for d in (train, dev) for a, b in zip(d.text1, d.text2)})
    frames = generate_frames(pairs, args.out / f"frames_track{t}.json", args.llm, args.load_4bit, args.gen_bs)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).float().to("cuda").eval()
    feats = {}
    for view, fr in (("orig", None), ("ellipsis", {}), ("llm", frames)):
        if view == "orig":
            tr, dv = train, dev
        else:
            (tr, ok_tr), (dv, ok_dv) = apply(train, fr), apply(dev, fr)
            print(f"{view}: valid frame rate train {ok_tr:.3f} dev {ok_dv:.3f}")
        feats[view] = (features(model, tok, tr, "cuda")[0], *features(model, tok, dv, "cuda"))
    llm_dev, _ = apply(dev, frames)
    llm_dev[["text1", "text2", "label"]].assign(a=dev.text1, b=dev.text2).head(25).to_csv(
        args.out / f"examples_track{t}.csv", index=False)

    systems = dict(feats)
    for v in ("ellipsis", "llm"):
        systems[f"orig+{v}"] = tuple(np.hstack([a, b]) for a, b in zip(feats["orig"], feats[v]))
    y = train.label.map(LABEL2ID)
    rows, preds = [], {}
    for name, (x_tr, x_dev, x_sw) in systems.items():
        clf = LogisticRegression(max_iter=3000).fit(x_tr, y)
        order = np.argsort(clf.classes_)
        preds[name] = self_swap(dev, clf.predict_proba(x_dev)[:, order], clf.predict_proba(x_sw)[:, order])
        r = predict_and_score(dev, preds[name], "dev", t, args.out, name)
        rows.append({"system": name, **{k: r[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}})
    cmp = [("orig+ellipsis", "orig+llm"), ("orig", "llm"), ("orig", "orig+llm")]
    cis = [{"a": a, "b": b, **paired_bootstrap(dev, preds[a], preds[b])} for a, b in cmp]
    pd.DataFrame(rows).to_csv(args.out / f"summary_track{t}.csv", index=False)
    (args.out / f"bootstrap_track{t}.json").write_text(json.dumps(cis, indent=2))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(cis).to_string(index=False))


if __name__ == "__main__":
    main()

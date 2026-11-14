"""Self-generated translation views with NLLB (per instance, no other test rows needed)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

NLLB = {"en": "eng_Latn", "es": "spa_Latn", "eu": "eus_Latn"}
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


def postprocess(src: str, hyp: str) -> str:
    """Match the phrase-style formatting of the source: NLLB tends to add a final period
    and capitalise the first letter of fragments like 'on a swing'."""
    hyp = hyp.strip()
    if hyp.endswith(".") and not src.rstrip().endswith("."):
        hyp = hyp[:-1].rstrip()
    if hyp and src[:1].islower() and hyp[:1].isupper():
        hyp = hyp[0].lower() + hyp[1:]
    return hyp


class Translator:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        )

    @torch.no_grad()
    def __call__(self, texts: list[str], src: str, tgt: str, batch_size: int = 64) -> list[str]:
        self.tok.src_lang = NLLB[src]
        out = []
        for s in range(0, len(texts), batch_size):
            chunk = texts[s : s + batch_size]
            batch = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(
                **batch,
                forced_bos_token_id=self.tok.convert_tokens_to_ids(NLLB[tgt]),
                num_beams=4,
                max_new_tokens=48,
            )
            hyps = self.tok.batch_decode(gen, skip_special_tokens=True)
            out.extend(postprocess(a, h) for a, h in zip(chunk, hyps))
        return out


def cached_translate(
    phrases: list[str], src: str, tgt: str, cache: Path, translator: Translator | None = None
) -> dict[str, str]:
    """Translate unique phrases once; results are stored in a JSON cache per direction."""
    path = cache / f"{src}2{tgt}.json"
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = sorted({p for p in phrases if p not in table})
    if todo:
        translator = translator or Translator()
        table.update(zip(todo, translator(todo, src, tgt)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")
    return table

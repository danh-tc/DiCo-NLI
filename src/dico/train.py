"""Fine-tune a cross-encoder and score it on dev.

    python -m dico.train --config configs/m0_deberta_base_en.yaml --seed 1
    python -m dico.train --config configs/m0_deberta_base_en.yaml --seed 1 --set epochs=3 lr=3e-5

Output: runs/<name>/seed<seed>/ with config.yaml, dev probs, predictions,
official scores and results.json. Same command locally and on Kaggle.
"""

from __future__ import annotations

import argparse
import inspect
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from . import env
from .data import LABEL2ID, LABELS, REV_PERM, add_reverse_index, load_split
from .decode import NEEDS_SWAP, correction_diagnostics, decode, swap_texts
from .evaluate import predict_and_score, save_json


def load_config(path: str, overrides: list[str]) -> dict:
    path = Path(path)
    cfg = yaml.safe_load((path.parent / "base.yaml").read_text(encoding="utf-8")) or {}
    cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    cfg.setdefault("name", path.stem)
    return cfg


class PairDataset(torch.utils.data.Dataset):
    """`template` (e.g. "Phrase A: {a}\\nPhrase B: {b}\\nRelation:") is for decoder LMs, which have
    no [SEP] pair encoding; without it, encoder models get the usual tokenizer(text1, text2)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int,
        template: str | None = None,
        pair_aware: bool = False,
        xling: str | None = None,
    ):
        if template:
            texts = [template.format(a=a, b=b) for a, b in zip(df.text1, df.text2)]
            self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        else:
            self.enc = tokenizer(list(df.text1), list(df.text2), truncation=True, max_length=max_len)
        self.rev_enc = None
        if pair_aware:
            if template:
                rev_texts = [template.format(a=b, b=a) for a, b in zip(df.text1, df.text2)]
                self.rev_enc = tokenizer(rev_texts, truncation=True, max_length=max_len)
            else:
                self.rev_enc = tokenizer(
                    list(df.text2), list(df.text1), truncation=True, max_length=max_len
                )
        self.labels = df.label.map(LABEL2ID).tolist() if "label" in df else None
        self.partners = cross_lingual_partners(df, xling) if xling else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.rev_enc is not None:
            item.update({f"rev_{k}": v[i] for k, v in self.rev_enc.items()})
        if self.partners is not None:
            # Another verified-translation view of the same source pair (train data only).
            options = self.partners[i]
            j, flip = options[random.randrange(len(options))] if options else (i, False)
            item.update({f"par_{k}": v[j] for k, v in self.enc.items()})
            item["par_flip"] = int(flip)
            item["par_mask"] = int(bool(options))
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def cross_lingual_partners(df: pd.DataFrame, mode: str) -> list[list[tuple[int, bool]]]:
    """For each row, the rows of the same source pair in another language combination.

    Orientation comes from the instance_id suffix (`original` = (phrase1, phrase2)), so a
    partner with the other suffix is the reversed relation (`flip=True`). Gold labels agree
    across languages after this alignment (checked on train and dev).
    mode="js":  any other language combination (mono and mixed).
    mode="en":  only the en-en view, used as a teacher; en-en rows get no partner.
    """
    if mode not in {"js", "en"}:
        raise ValueError(f"xling_mode must be 'js' or 'en', got {mode!r}")
    df = df.reset_index(drop=True)
    flipped = df.instance_id.str.endswith("__flipped").to_numpy()
    langs = list(zip(df.text1_lang, df.text2_lang))
    by_pair: dict[str, list[int]] = {}
    for i, pid in enumerate(df.pair_id):
        by_pair.setdefault(pid, []).append(i)
    out = []
    for i, pid in enumerate(df.pair_id):
        if mode == "en":
            cand = [] if langs[i] == ("en", "en") else [
                j for j in by_pair[pid] if langs[j] == ("en", "en")]
        else:
            cand = [j for j in by_pair[pid] if langs[j] != langs[i]]
        out.append([(j, bool(flipped[i] != flipped[j])) for j in cand])
    return out


class PairAwareCollator:
    """Pad normal, swapped (rev_) and partner (par_) inputs together so all share a shape."""

    PREFIXES = ("rev_", "par_")

    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        batch = {"labels": torch.tensor([row["labels"] for row in features], dtype=torch.long)}
        for flag in ("par_flip", "par_mask"):
            if flag in features[0]:
                batch[flag] = torch.tensor([row[flag] for row in features], dtype=torch.bool)
        skip = {"labels", "par_flip", "par_mask"}
        groups = [[{k: v for k, v in row.items() if k not in skip and not k.startswith(self.PREFIXES)}
                   for row in features]]
        prefixes = [p for p in self.PREFIXES if any(k.startswith(p) for k in features[0])]
        for p in prefixes:
            groups.append([{k[len(p):]: v for k, v in row.items() if k.startswith(p) and k not in skip}
                           for row in features])
        size = len(features)
        combined = self.base([row for group in groups for row in group])
        for g, prefix in enumerate([""] + prefixes):
            batch.update({f"{prefix}{k}": v[g * size:(g + 1) * size] for k, v in combined.items()})
        return batch


def pair_aware_loss(normal_logits, reverse_logits, labels, weight: float, label_smoothing: float):
    """Supervise both directions and align their distributions with Rev()."""
    perm = torch.as_tensor(REV_PERM, device=labels.device)
    reverse_labels = perm[labels]
    supervised = 0.5 * (
        F.cross_entropy(normal_logits, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(reverse_logits, reverse_labels, label_smoothing=label_smoothing)
    )
    aligned_reverse = reverse_logits[:, REV_PERM]
    p_log = F.log_softmax(normal_logits, dim=-1)
    q_log = F.log_softmax(aligned_reverse, dim=-1)
    p, q = p_log.exp(), q_log.exp()
    middle = 0.5 * (p + q)
    js = 0.5 * (
        F.kl_div(middle.log(), p, reduction="batchmean")
        + F.kl_div(middle.log(), q, reduction="batchmean")
    )
    return supervised + weight * js


def xling_loss(normal_logits, partner_logits, flip, mask, mode: str):
    """Cross-lingual agreement on the same source pair, with Rev() for flipped partners.

    js: symmetric Jensen-Shannon between the two views (both receive gradient).
    en: KL(teacher || student) with the en-en view as a frozen teacher.
    Rows without a partner (mask=False) contribute nothing.
    """
    if not mask.any():
        return normal_logits.new_zeros(())
    perm = torch.as_tensor(REV_PERM, device=normal_logits.device)
    aligned = torch.where(flip[:, None], partner_logits[:, perm], partner_logits)
    p_log = F.log_softmax(normal_logits[mask], dim=-1)
    q_log = F.log_softmax(aligned[mask], dim=-1)
    if mode == "en":
        q_log = q_log.detach()
        return F.kl_div(p_log, q_log, log_target=True, reduction="batchmean")
    p, q = p_log.exp(), q_log.exp()
    m_log = (0.5 * (p + q)).clamp_min(1e-12).log()
    return 0.5 * (F.kl_div(m_log, p, reduction="batchmean") + F.kl_div(m_log, q, reduction="batchmean"))


class PairAwareTrainer(Trainer):
    def __init__(
        self,
        *args,
        pair_consistency_weight: float,
        xling_weight: float = 0.0,
        xling_mode: str = "js",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pair_consistency_weight = pair_consistency_weight
        self.xling_weight = xling_weight
        self.xling_mode = xling_mode

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        flip, mask = inputs.pop("par_flip", None), inputs.pop("par_mask", None)
        views = [{k: v for k, v in inputs.items() if not k.startswith(("rev_", "par_"))}]
        for prefix in ("rev_", "par_"):
            part = {k[len(prefix):]: v for k, v in inputs.items() if k.startswith(prefix)}
            if part:
                views.append(part)
        combined = {k: torch.cat([view[k] for view in views], dim=0) for k in views[0]}
        outputs = model(**combined)
        logits = outputs.logits.chunk(len(views), dim=0)
        loss = pair_aware_loss(
            logits[0],
            logits[1],
            labels,
            self.pair_consistency_weight,
            self.args.label_smoothing_factor,
        )
        if len(views) == 3 and self.xling_weight > 0:
            loss = loss + self.xling_weight * xling_loss(logits[0], logits[2], flip, mask, self.xling_mode)
        outputs["logits"] = logits[0]
        return (loss, outputs) if return_outputs else loss


def bientail_scores(logits_ab: torch.Tensor, logits_ba: torch.Tensor, entail_index: int) -> torch.Tensor:
    """4-label log-probs from two directional entailment probabilities.

    e1 = P(A => B), e2 = P(B => A) from the NLI head's entailment class. The four combinations
    (EQ, FWD, BWD, NEG) = (e1 e2, e1 (1-e2), (1-e1) e2, (1-e1)(1-e2)) sum to one, and swapping
    the inputs swaps FWD/BWD exactly, so the decoder is reversal-consistent by construction.
    """
    le1 = F.log_softmax(logits_ab.float(), -1)[:, entail_index]
    le2 = F.log_softmax(logits_ba.float(), -1)[:, entail_index]
    ln1 = torch.log1p(-le1.exp().clamp(max=1 - 1e-6))
    ln2 = torch.log1p(-le2.exp().clamp(max=1 - 1e-6))
    return torch.stack([le1 + le2, le1 + ln2, ln1 + le2, ln1 + ln2], dim=-1)  # LABELS order


class BiEntailTrainer(Trainer):
    """Train e(X => Y) on both directions; targets come from the 4-label table."""

    def __init__(self, *args, entail_index: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.entail_index = entail_index

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        normal = {k: v for k, v in inputs.items() if not k.startswith("rev_")}
        reverse = {k[4:]: v for k, v in inputs.items() if k.startswith("rev_")}
        combined = {k: torch.cat([normal[k], reverse[k]], dim=0) for k in normal}
        outputs = model(**combined)
        logits_ab, logits_ba = outputs.logits.chunk(2, dim=0)
        scores = bientail_scores(logits_ab, logits_ba, self.entail_index)
        # Joint log-likelihood of the gold cell = BCE on both directions.
        loss = F.nll_loss(scores, labels)
        outputs["logits"] = scores  # softmax(scores) recovers the 4 cell probabilities
        return (loss, outputs) if return_outputs else loss


def uses_pair_inputs(cfg: dict) -> bool:
    return cfg.get("method", "m0") == "bientail" or float(cfg.get("pair_consistency_weight", 0.0)) > 0


def trainer_setup(cfg: dict, model) -> tuple[type, dict]:
    if cfg.get("method", "m0") == "bientail":
        return BiEntailTrainer, {"entail_index": model.config.dico_entail_index}
    if float(cfg.get("pair_consistency_weight", 0.0)) > 0:
        return PairAwareTrainer, pair_trainer_kwargs(cfg)
    return Trainer, {}


def xling_setting(cfg: dict, pair_aware: bool) -> str | None:
    """Cross-lingual partner mode for training, or None when disabled."""
    if float(cfg.get("xling_consistency_weight", 0.0)) <= 0:
        return None
    if not pair_aware:
        raise ValueError("xling_consistency_weight needs pair_consistency_weight > 0")
    return cfg.get("xling_mode", "js")


def pair_trainer_kwargs(cfg: dict) -> dict:
    return {
        "pair_consistency_weight": float(cfg["pair_consistency_weight"]),
        "xling_weight": float(cfg.get("xling_consistency_weight", 0.0)),
        "xling_mode": cfg.get("xling_mode", "js"),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {"weighted_f1": f1_score(labels, preds, average="weighted")}


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def build_model(cfg: dict, tokenizer, prec: str):
    if cfg.get("method", "m0") == "bientail":
        # Keep the pretrained NLI head: its entailment logit is e(X => Y).
        model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"]).float()
        label2id = {k.lower(): v for k, v in model.config.label2id.items()}
        if "entailment" not in label2id:
            raise ValueError(f"bientail needs an NLI checkpoint with an 'entailment' label: {cfg['model_name']}")
        model.config.dico_method = "bientail"
        model.config.dico_entail_index = int(label2id["entailment"])
        return model
    kwargs = dict(
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
        ignore_mismatched_sizes=cfg.get("ignore_mismatched_sizes", False),
    )
    lora = cfg.get("lora")
    if not lora:
        # transformers 5 loads the checkpoint dtype (fp16 for deberta-v3) -> NaN with AdamW.
        # Master weights must be fp32; bf16/fp16 in TrainingArguments is autocast only.
        return AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], **kwargs).float()

    from peft import LoraConfig, TaskType, get_peft_model

    # Frozen base in half precision (fits 8 GB); trainable params (LoRA + score head) in fp32.
    base_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec, torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], dtype=base_dtype, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,  # also keeps the new `score` head trainable
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get("target_modules", "all-linear"),
    ))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp16 grads would break the GradScaler on T4
    model.print_trainable_parameters()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if cfg.get("method", "m0") not in {"m0", "bientail"}:
        raise NotImplementedError(f"method {cfg['method']} not implemented yet")
    env.set_seed(args.seed)
    prec = env.precision()
    run_dir = env.runs_dir() / cfg["name"] / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": args.seed}), encoding="utf-8")
    print(env.describe())
    print(f"run_dir={run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(cfg, tokenizer, prec)

    template = cfg.get("template")
    pair_aware = uses_pair_inputs(cfg)
    xling = xling_setting(cfg, pair_aware)
    train_df = load_split("train", cfg["train_tracks"])
    dev_sets = {t: add_reverse_index(load_split("dev", [t])) for t in cfg["eval_tracks"]}
    if cfg.get("augment_transitive"):
        from .augment import augment, text_pairs

        train_df = augment(train_df, exclude=set().union(*(text_pairs(d) for d in dev_sets.values())))
    train_ds = PairDataset(train_df, tokenizer, cfg["max_len"], template, pair_aware, xling)
    dev_ds = {
        t: PairDataset(df, tokenizer, cfg["max_len"], template, pair_aware)
        for t, df in dev_sets.items()
    }

    # transformers 5 dropped warmup_ratio (warmup_steps takes a float ratio); 4.x (Kaggle) still has it.
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        warmup = {"warmup_ratio": cfg["warmup_ratio"]}
    else:
        warmup = {"warmup_steps": float(cfg["warmup_ratio"])}
    targs = TrainingArguments(
        **warmup,
        output_dir=str(run_dir / "ckpt"),
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        weight_decay=cfg["weight_decay"],
        label_smoothing_factor=cfg.get("label_smoothing", 0.0),
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="no",  # fixed epochs, no best-on-dev selection: dev scores stay honest
        logging_steps=50,
        bf16=prec == "bf16",
        fp16=prec == "fp16",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=not pair_aware,
    )
    trainer_cls, trainer_kwargs = trainer_setup(cfg, model)
    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds[cfg["eval_tracks"][0]],
        data_collator=PairAwareCollator(tokenizer) if pair_aware else DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        **trainer_kwargs,
    )

    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    save_json(trainer.state.log_history, run_dir / "log_history.json")

    results = []
    need_swap = any(name in NEEDS_SWAP for name in cfg["decoders"])
    for track, df in dev_sets.items():
        probs = softmax(trainer.predict(dev_ds[track]).predictions)
        np.save(run_dir / f"dev_track{track}_probs.npy", probs)
        probs_swap = None
        if need_swap:  # p(.|B,A) for every row, computed by us: no twin lookup
            swap_ds = PairDataset(swap_texts(df), tokenizer, cfg["max_len"], template, pair_aware)
            probs_swap = softmax(trainer.predict(swap_ds).predictions)
            np.save(run_dir / f"dev_track{track}_swap_probs.npy", probs_swap)
        raw = decode("argmax", df, probs, cfg)
        for name in cfg["decoders"]:
            labels = decode(name, df, probs, cfg, probs_swap)
            row = predict_and_score(df, labels, "dev", track, run_dir, name)
            row.update(correction_diagnostics(df, raw, labels))
            results.append({"name": cfg["name"], "seed": args.seed, **row})

    save_json({"train_minutes": round(train_minutes, 2), "precision": prec, "results": results},
              run_dir / "results.json")
    if cfg.get("keep_model"):
        trainer.save_model(str(run_dir / "model"))
        tokenizer.save_pretrained(str(run_dir / "model"))
    print(pd.DataFrame(results).drop(columns=["name"]).round(4).to_string(index=False))
    print(f"train time: {train_minutes:.1f} min")


if __name__ == "__main__":
    main()

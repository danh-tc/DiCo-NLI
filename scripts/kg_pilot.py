"""KG pilot: how much of M0's error is lexical substitution that WordNet could explain?

For each dev instance we diff the (lemmatised) token sets of A and B and bucket the pair:
  same      no content difference after lemmatising (e.g. law / laws)
  add       one side only adds words (modifier insertion) -- surface-solvable
  subst     both sides have their own words -> look up a WordNet relation between them
  (subst is split by the strongest relation found: syn, hypo (A word is-a B word), hyper,
   ant, cohypo, none, oov)

Then we cross the buckets with M0 errors and ask: how many errors fall in subst-with-relation,
and does the WordNet direction predict the gold label exactly where M0 is wrong?

    python scripts/kg_pilot.py --track 1 --run runs/m0_deberta_base_en --decoder joint
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dico import env  # noqa: E402

WN_LANG = {"en": "eng", "es": "spa", "eu": "eus"}
STOP = {
    "en": set("a an the of in on at to for with and or by from 's is are be this that its his her their".split()),
    "es": set("el la los las un una unos unas de del en a al con por para y o que su sus es se".split()),
    "eu": set("eta da dira bat edo ere ez du".split()),
}
TOKEN = re.compile(r"\w+(?:-\w+)*", re.UNICODE)

# label a WordNet relation implies for (A, B); None = no directional prediction
REL_LABEL = {
    "syn": "EQUIVALENCE",
    "hypo": "FORWARD_ENTAILMENT",   # A word is-a B word: "dog" vs "animal" -> A => B
    "hyper": "BACKWARD_ENTAILMENT",
    "ant": "NEGATIVE_OTHER",
    "cohypo": "NEGATIVE_OTHER",
}
REL_ORDER = ["syn", "hypo", "hyper", "ant", "cohypo", "none", "oov"]


def synsets(word: str, lang: str):
    if lang == "en":
        base = wn.morphy(word) or word
        return wn.synsets(base)
    return wn.synsets(word, lang=WN_LANG[lang])


def lemma(word: str, lang: str) -> str:
    if lang == "en":
        return wn.morphy(word) or word
    return word  # no lemmatiser for es/eu here (pilot): OOV rate will show the cost


def content_tokens(text: str, lang: str) -> set[str]:
    return {lemma(t, lang) for t in TOKEN.findall(text.lower()) if t not in STOP[lang]}


def hyper_closure(ss) -> set:
    return set(ss.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))


def relation(wa: str, la: str, wb: str, lb: str) -> str:
    sa, sb = synsets(wa, la), synsets(wb, lb)
    if not sa or not sb:
        return "oov"
    sa_set, sb_set = set(sa), set(sb)
    if sa_set & sb_set:
        return "syn"
    if any(b in hyper_closure(a) for a in sa for b in sb):
        return "hypo"
    if any(a in hyper_closure(b) for a in sa for b in sb):
        return "hyper"
    ant = {an.synset() for a in sa for l in a.lemmas() for an in l.antonyms()}
    if ant & sb_set:
        return "ant"
    pa = {h for a in sa for h in a.hypernyms()}
    if any(h in pa for b in sb for h in b.hypernyms()):
        return "cohypo"
    return "none"


def bucket(row) -> tuple[str, str]:
    la, lb = row.text1_lang, row.text2_lang
    ta, tb = content_tokens(row.text1, la), content_tokens(row.text2, lb)
    if la != lb:  # mixed-language pair: token diff is meaningless, compare all pairs
        only_a, only_b = ta, tb
    else:
        only_a, only_b = ta - tb, tb - ta
    if not only_a and not only_b:
        return ("same" if ta else "empty"), ""
    if not only_a or not only_b:
        return "add", ""
    rels = {relation(a, la, b, lb) for a in only_a for b in only_b}
    best = next(r for r in REL_ORDER if r in rels)
    return "subst", best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=1)
    ap.add_argument("--run", type=Path, default=ROOT / "runs/m0_deberta_base_en")
    ap.add_argument("--decoder", default="joint")
    args = ap.parse_args()

    gold = pd.read_csv(env.data_dir() / "dev" / f"dico_nli_dev_track{args.track}_participant_labeled.csv",
                       keep_default_na=False)
    b = gold.apply(bucket, axis=1, result_type="expand")
    gold["bucket"], gold["rel"] = b[0], b[1]
    gold["cat"] = gold.bucket.where(gold.bucket != "subst", "subst:" + gold.rel)

    seed_dirs = sorted(args.run.glob("seed*"))
    err_cols = []
    for sd in seed_dirs:
        f = sd / f"dev_track{args.track}_{args.decoder}.csv"
        if not f.exists():
            continue
        pred = pd.read_csv(f, keep_default_na=False).set_index("instance_id")["label"]
        col = f"err_{sd.name}"
        gold[col] = gold.instance_id.map(pred) != gold.label
        err_cols.append(col)
    if not err_cols:
        sys.exit(f"no predictions under {args.run}")
    gold["err"] = gold[err_cols].mean(axis=1)  # error rate averaged over seeds

    n, n_err = len(gold), gold.err.sum()
    print(f"track {args.track}: {n} dev instances, M0 {args.decoder} over {len(err_cols)} seeds, "
          f"mean errors {n_err:.1f} ({n_err / n:.1%})\n")

    tab = gold.groupby("cat").agg(n=("err", "size"), err_rate=("err", "mean"), errors=("err", "sum"))
    tab["share_of_instances"] = tab.n / n
    tab["share_of_errors"] = tab.errors / n_err
    order = ["empty", "same", "add"] + [f"subst:{r}" for r in REL_ORDER]
    tab = tab.reindex([c for c in order if c in tab.index])
    print(tab.round(3).to_string(), "\n")

    # Headroom: where WordNet makes a directional prediction, is it right, and is it right where M0 is wrong?
    g = gold[gold.rel.isin(REL_LABEL)].copy()
    g["wn_pred"] = g.rel.map(REL_LABEL)
    g["wn_ok"] = g.wn_pred == g.label
    wrong = g[g.err > 0]
    print(f"subst with a WordNet relation: {len(g)} instances ({len(g) / n:.1%}), "
          f"rule accuracy {g.wn_ok.mean():.1%}")
    print(f"  of which M0 errs on (weighted): {wrong.err.sum():.1f} errors = "
          f"{wrong.err.sum() / n_err:.1%} of all M0 errors")
    print(f"  WordNet rule right on those errors: {(wrong.err * wrong.wn_ok).sum():.1f} "
          f"= {(wrong.err * wrong.wn_ok).sum() / n_err:.1%} of all M0 errors  <- KG headroom (upper bound)\n")
    print("rule accuracy by relation:")
    print(g.groupby("rel").agg(n=("wn_ok", "size"), acc=("wn_ok", "mean")).round(3).to_string(), "\n")
    print("gold label x category (counts):")
    print(pd.crosstab(gold.cat, gold.label).reindex(tab.index).to_string(), "\n")

    ex = gold[(gold.err > 0) & gold.cat.str.startswith("subst")].sort_values("err", ascending=False)
    print("sample M0 errors in subst buckets:")
    for r in ex.head(15).itertuples():
        print(f"  [{r.cat:13s}] {r.text1!r:40s} | {r.text2!r:40s} gold={r.label}")

    out = ROOT / "runs" / "kg_pilot"
    out.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out / f"track{args.track}_{args.run.name}_{args.decoder}.csv", index=False)


if __name__ == "__main__":
    main()

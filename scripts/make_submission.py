"""Validate prediction CSVs against the official submission templates and pack a Codabench ZIP.

    # dev phase, e.g. from one run:
    python scripts/make_submission.py --split dev \
        --track1 runs/m0_deberta_base_en/seed1/dev_track1_joint.csv \
        --out submissions/dev_m0_joint.zip

    # test phase (templates released later): point --template_dir at the test templates
    python scripts/make_submission.py --split test --template_dir path/to/test \
        --track1 a.csv --track2 b.csv --track3 c.csv --track4 d.csv --out submissions/test_final.zip

Rules enforced (Codabench page): files named trackN_predictions.csv at the ZIP root, header exactly
`instance_id,label`, one row per template id (no missing / duplicate / unknown ids), valid labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dico import env  # noqa: E402
from dico.data import LABELS  # noqa: E402


def template_ids(split: str, track: int, template_dir: Path | None) -> list[str]:
    name = f"dico_nli_{split}_track{track}_submission_template.csv"
    path = (template_dir or env.data_dir() / split) / name
    if not path.exists():
        sys.exit(f"template not found: {path} (use --template_dir)")
    with path.open(encoding="utf-8", newline="") as f:
        return [row["instance_id"] for row in csv.DictReader(f)]


def load_and_check(pred_path: Path, ids: list[str], track: int) -> list[tuple[str, str]]:
    with pred_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    errors = []
    if header != ["instance_id", "label"]:
        errors.append(f"header must be exactly instance_id,label (got {header})")
    pred = {}
    for r in rows:
        if len(r) != 2:
            errors.append(f"bad row {r}")
            continue
        iid, label = r
        if iid in pred:
            errors.append(f"duplicate id {iid}")
        if label not in LABELS:
            errors.append(f"invalid label {label!r} for {iid}")
        pred[iid] = label
    missing = set(ids) - pred.keys()
    unknown = pred.keys() - set(ids)
    if missing:
        errors.append(f"{len(missing)} missing ids, e.g. {sorted(missing)[:3]}")
    if unknown:
        errors.append(f"{len(unknown)} unknown ids, e.g. {sorted(unknown)[:3]}")
    if errors:
        sys.exit(f"track{track} ({pred_path}) INVALID:\n  " + "\n  ".join(errors[:10]))
    return [(iid, pred[iid]) for iid in ids]  # template order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--template_dir", type=Path)
    for t in range(1, 5):
        ap.add_argument(f"--track{t}", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tracks = {t: getattr(args, f"track{t}") for t in range(1, 5) if getattr(args, f"track{t}")}
    if not tracks:
        sys.exit("give at least one --trackN file")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for t, path in sorted(tracks.items()):
            rows = load_and_check(path, template_ids(args.split, t, args.template_dir), t)
            body = "instance_id,label\n" + "".join(f"{i},{l}\n" for i, l in rows)
            zf.writestr(f"track{t}_predictions.csv", body)  # flat: no directories
            print(f"track{t}: {len(rows)} rows OK")

    with zipfile.ZipFile(args.out) as zf:
        names = zf.namelist()
    print(f"\n{args.out} contains: {names}")
    if len(tracks) < 4:
        print("note: fewer than 4 tracks -> macro4_* scores will show -1.0 (does not hurt per-track standing)")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Clone (or update) the organizers' repo: data + official scorer -> external/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/external/SemEval-2027-Task-2-DiCo-NLI"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --ff-only
else
  mkdir -p "$ROOT/external"
  git clone --depth 1 https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git "$DEST"
fi
ls "$DEST/final_data/train" | head

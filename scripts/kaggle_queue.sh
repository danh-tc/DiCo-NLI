#!/usr/bin/env bash
# Run a list of Qwen jobs on one GPU (Kaggle 2xT4: start one queue per GPU, in parallel).
# One process per GPU keeps the single-GPU batch size of the config (no DataParallel).
#
#   bash scripts/kaggle_queue.sh GPU JOB [JOB ...]
#   JOB = train:<lang>:<seed> | oof:<lang>          lang in en, es, eu
#
# Each job logs to $RUNS/logs/<gpu>_<job>.log; a failed job does not stop the queue.
# Progress: $RUNS/logs/status.txt.
set -u
GPU=$1
shift
cd "$(dirname "$0")/.."
RUNS=${DICO_RUNS_DIR:-/kaggle/working/runs}
LOG=$RUNS/logs
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src
CFG=configs/qwen_cls_en.yaml
declare -A TRACK=([en]=1 [es]=2 [eu]=3)

for job in "$@"; do
  IFS=: read -r kind lang seed <<< "$job"
  t=${TRACK[$lang]}
  name=qwen_cls_$lang
  over=(name=$name "train_tracks=[$t]" "eval_tracks=[$t]" "decoders=[argmax,self_swap]")
  if [ "$kind" = train ]; then
    cmd=(python -m dico.train --config $CFG --seed "$seed" --set "${over[@]}")
  else
    cmd=(python scripts/generate_oof.py --config $CFG --folds 3 --overwrite
         --out "$RUNS/oof/$name" --set "${over[@]}")
  fi
  tag=${GPU}_${job//:/_}
  echo "$(date +%T) START $tag" | tee -a "$LOG/status.txt"
  if "${cmd[@]}" > "$LOG/$tag.log" 2>&1; then
    echo "$(date +%T) OK    $tag" | tee -a "$LOG/status.txt"
  else
    echo "$(date +%T) FAIL  $tag" | tee -a "$LOG/status.txt"
  fi
done

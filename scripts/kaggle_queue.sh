#!/usr/bin/env bash
# Run a list of jobs on one GPU (Kaggle 2xT4: start one queue per GPU, in parallel).
# One process per GPU keeps the single-GPU batch size of the config (no DataParallel).
#
#   bash scripts/kaggle_queue.sh GPU JOB [JOB ...]
#   JOB = train:<target>:<seed> | oof:<target>
#   target = en | es | eu   Qwen2.5-1.5B LoRA on that track only (configs/qwen_cls_en.yaml)
#          | main           main system, 4 tracks (configs/m0_mdeberta_xnli_pair_all.yaml, as is)
#          | <config>       any configs/<config>.yaml, as is (e.g. qwen_cls_all)
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
declare -A TRACK=([en]=1 [es]=2 [eu]=3)

for job in "$@"; do
  IFS=: read -r kind target seed <<< "$job"
  if [ "$target" = main ]; then
    CFG=configs/m0_mdeberta_xnli_pair_all.yaml
    name=m0_mdeberta_xnli_pair_all
    over=("decoders=[argmax,self_swap]")
  elif [ -f "configs/$target.yaml" ]; then
    CFG=configs/$target.yaml
    name=$target
    over=("decoders=[argmax,self_swap]")
  else
    CFG=configs/qwen_cls_en.yaml
    t=${TRACK[$target]}
    name=qwen_cls_$target
    over=(name=$name "train_tracks=[$t]" "eval_tracks=[$t]" "decoders=[argmax,self_swap]")
  fi
  # KEEP_MODEL=1: save runs/<name>/seed<k>/model for test prediction (scripts/predict_final.py)
  [ "${KEEP_MODEL:-0}" = 1 ] && over+=(keep_model=true)
  # Variants: SUFFIX is appended to the run name, EXTRA holds more key=value overrides,
  # e.g. SUFFIX=_aug EXTRA="augment_transitive=true"
  if [ -n "${SUFFIX:-}" ]; then name=$name$SUFFIX; over+=(name=$name); fi
  # shellcheck disable=SC2206
  [ -n "${EXTRA:-}" ] && over+=($EXTRA)
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

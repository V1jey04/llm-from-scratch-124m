#!/usr/bin/env bash
# One Kaggle training session. Resumes automatically when an earlier session's
# output (containing RUN/checkpoint.pt) is attached to the notebook as an input.
#
#   bash kaggle/train.sh configs/main_124m.json run_main [KEY=VALUE ...]
#
# Extra KEY=VALUE settings go to `lm.train --set`, e.g. time_limit_hours=6.5.
# KAGGLE_INPUT, KAGGLE_WORKING, NPROC and PYTHON can be overridden for local tests.
set -euo pipefail

CONFIG=$1
RUN=$2
shift 2
INPUT=${KAGGLE_INPUT:-/kaggle/input}
WORKING=${KAGGLE_WORKING:-/kaggle/working}
NPROC=${NPROC:-2}
PYTHON=${PYTHON:-python}

MANIFEST=$(find -L "$INPUT" -name manifest.json | head -n 1)
if [ -z "$MANIFEST" ]; then
  echo "No token data (manifest.json) found under $INPUT; attach the data notebook's output." >&2
  exit 1
fi
DATA=$(dirname "$MANIFEST")
OUT=$WORKING/$RUN
mkdir -p "$OUT"

if [ ! -f "$OUT/checkpoint.pt" ]; then
  FOUND=$(find -L "$INPUT" -path "*/$RUN/checkpoint.pt")
  if [ -n "$FOUND" ]; then
    if [ "$(printf '%s\n' "$FOUND" | wc -l)" -gt 1 ]; then
      echo "Several checkpoints for $RUN are attached; attach only the latest session's output:" >&2
      printf '%s\n' "$FOUND" >&2
      exit 1
    fi
    echo "Resuming from $FOUND"
    for name in checkpoint.pt best_model.pt log.jsonl; do
      if [ -f "$(dirname "$FOUND")/$name" ]; then
        cp "$(dirname "$FOUND")/$name" "$OUT/"
      fi
    done
  else
    echo "No earlier checkpoint for $RUN attached: starting from scratch."
  fi
fi

echo "data: $DATA"
echo "out:  $OUT"
if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv
fi
if [ "$NPROC" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NPROC" -m lm.train --config "$CONFIG" \
    --set data_dir="$DATA" out_dir="$OUT" "$@"
else
  "$PYTHON" -m lm.train --config "$CONFIG" --set data_dir="$DATA" out_dir="$OUT" "$@"
fi
ls -la "$OUT"
tail -n 3 "$OUT/log.jsonl"

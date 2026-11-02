#!/usr/bin/env bash
# Instruction-tune the pretrained model on Kaggle's 2×T4 (under an hour).
# Attach the training notebook's output (containing run_main/final_model.pt).
#
#   bash kaggle/finetune.sh [KEY=VALUE ...]
#
# The chat data is downloaded and tokenized here (under a minute), so only the
# model needs attaching. Internet must be on.
set -euo pipefail

INPUT=${KAGGLE_INPUT:-/kaggle/input}
WORKING=${KAGGLE_WORKING:-/kaggle/working}
NPROC=${NPROC:-2}

FOUND=$(find -L "$INPUT" -path "*/run_main/final_model.pt")
if [ -z "$FOUND" ] || [ "$(printf '%s\n' "$FOUND" | wc -l)" -gt 1 ]; then
  echo "Attach exactly one output containing run_main/final_model.pt; found:" >&2
  printf '%s\n' "$FOUND" >&2
  exit 1
fi
DATA=/tmp/sft_data
OUT=$WORKING/run_sft
echo "start: $FOUND"

pip install -q pyarrow huggingface_hub
python scripts/prepare_sft.py --tokenizer tokenizers/fineweb_edu_32k.json --out "$DATA" --cache /tmp/hf

if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv
fi
SETTINGS=(data_dir="$DATA" out_dir="$OUT" init_from="$FOUND" "$@")
if [ "$NPROC" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NPROC" -m lm.train --config configs/sft_124m.json --set "${SETTINGS[@]}"
else
  python -m lm.train --config configs/sft_124m.json --set "${SETTINGS[@]}"
fi
ls -la "$OUT"
tail -n 3 "$OUT/log.jsonl"

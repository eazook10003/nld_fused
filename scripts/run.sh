#!/bin/bash
# Full pipeline for one benchmark on one GPU: cost table -> timing (baseline vs final method) -> FLOP records -> table.
#   bash scripts/run.sh gsm8k runs/gsm8k
#   bash scripts/run.sh math500 runs/math500
# Environment variables (optional):
#   REPEATS  number of passes over the whole set (default 6; the first is dropped by analyze.py)
#   LIMIT    first N problems only, e.g. LIMIT=20 for a quick test (default: all)
#   NLD_MODEL  HF repo id or local checkpoint directory (default nvidia/Nemotron-Labs-Diffusion-8B)
set -e
DATASET=$1
OUT=$2
REPEATS=${REPEATS:-6}
if [ -z "$DATASET" ] || [ -z "$OUT" ]; then
    echo "usage: bash scripts/run.sh gsm8k|math500 OUTDIR"; exit 1
fi
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

if [ "$DATASET" = "gsm8k" ]; then
    CAP=8192; THINK=""
elif [ "$DATASET" = "math500" ]; then
    CAP=16000; THINK="--thinking"
else
    echo "unknown dataset $DATASET"; exit 1
fi
LIM=""
if [ -n "$LIMIT" ]; then LIM="--limit $LIMIT"; fi

hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 1) cost table on this GPU
python cost_table.py --out "$OUT/cost_table"
DYN=$(cat "$OUT/cost_table.txt")
echo "cost table: $DYN"

# 2) timing + accuracy, both arms alternating on this GPU
python bench.py --dataset "$DATASET" $THINK $LIM --arms upstream,dynfat --batch 20 --repeats "$REPEATS" \
    --max-new-tokens "$CAP" --dyn "$DYN" --out "$OUT/timing"

# 3) FLOP records (untimed regeneration)
python flops_record.py --dataset "$DATASET" $THINK $LIM --arm upstream --max-new-tokens "$CAP" --out "$OUT/flops_upstream"
python flops_record.py --dataset "$DATASET" $THINK $LIM --arm dynfat --max-new-tokens "$CAP" --dyn "$DYN" --out "$OUT/flops_dynfat"

# 4) table
python analyze.py "$OUT"
date

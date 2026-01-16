#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mnt/usercache/huggingface/Qwen2.5-VL-7B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/userdata/VisualReasoning/data/train/stage1/gsm8k.jsonl}"
OUT_DIR="${OUT_DIR:-/mnt/userdata/VisualReasoning/SFT/out_stage1_dynamic}"

# Balanced plan: dynamic latent size range [32, 256]
LATENT_SIZE="${LATENT_SIZE:-256}"           # Maximum latent token count
MIN_LATENT_SIZE="${MIN_LATENT_SIZE:-32}"     # Minimum latent token count
EPOCHS="${EPOCHS:-1}"
BSZ="${BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

uv run -m SFT.main \
  --stage stage1 \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUT_DIR" \
  --latent_size "$LATENT_SIZE" \
  --min_latent_size "$MIN_LATENT_SIZE" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --epochs "$EPOCHS" --bsz "$BSZ" --grad_accum_steps "$GRAD_ACCUM"



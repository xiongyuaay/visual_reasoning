#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mnt/userdata/VisualReasoning/SFT/out_stage1_dynamic}"
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/userdata/VisualReasoning/data/train/stage1/gsm8k.jsonl}"
TEACHER_LATENT_DIR="${TEACHER_LATENT_DIR:-/mnt/userdata/VisualReasoning/SFT/teacher_latents_v4}"

K_MAX="${K_MAX:-256}"  # Balanced plan: 256 to preserve more information
MAX_SEQ_LEN="${MAX_SEQ_LEN:-3072}"  # 降低序列长度，因为图片已resize
BATCH_SIZE="${BATCH_SIZE:-4}"       # 增加batch size，因为显存充足
NUM_SHARDS="${NUM_SHARDS:-4}"       # 匹配3张A100
MAX_SAMPLES="${MAX_SAMPLES:--1}"

mkdir -p "$TEACHER_LATENT_DIR"
LOG_DIR="${LOG_DIR:-$TEACHER_LATENT_DIR/logs_precompute}"
mkdir -p "$LOG_DIR"

for ((i=0; i<NUM_SHARDS; i++)); do
  (
    CUDA_VISIBLE_DEVICES="$i" uv run python -m SFT.precompute_teacher_latents \
      --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --train_jsonl "$TRAIN_JSONL" \
      --output_dir "$TEACHER_LATENT_DIR" \
      --k_max "$K_MAX" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --batch_size "$BATCH_SIZE" \
      --num_shards "$NUM_SHARDS" \
      --shard_id "$i" \
      --max_samples "$MAX_SAMPLES"
  ) |& tee "$LOG_DIR/shard_${i}.log" &
done
wait


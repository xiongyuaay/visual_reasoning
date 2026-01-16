#!/usr/bin/env bash
set -euo pipefail

# Verify that stage3 model generates latent tokens during inference

MODEL_PATH="${MODEL_PATH:-/netdisk/xiongyuan/VisualReasoning/SFT/out_stage3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"

echo "Verifying latent token generation for model: $MODEL_PATH"
echo ""

uv run python verify_latent_output.py \
  --model_path "$MODEL_PATH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --num_samples "$NUM_SAMPLES"

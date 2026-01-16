#!/usr/bin/env bash
set -euo pipefail

# GPU configuration: specify which GPUs to use (e.g., "2,3" for GPU 2 and 3)
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,7,8}"

# DeepSpeed configuration: disable JIT compilation (runtime image has no nvcc)
export DS_BUILD_OPS=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_UTILS=0
export DS_SKIP_CUDA_CHECK=1

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mnt/userdata/VisualReasoning/SFT/out_stage1_dynamic}"   # 训练用的初始模型
TRAIN_JSONL="${TRAIN_JSONL:-/mnt/userdata/VisualReasoning/data/train/stage1/gsm8k.jsonl}"  # 训练数据
TEACHER_LATENT_DIR="${TEACHER_LATENT_DIR:-/mnt/userdata/VisualReasoning/SFT/teacher_latents_v4}" # 离线 teacher latents
OUT_DIR="${OUT_DIR:-/mnt/userdata/VisualReasoning/SFT/out_stage2_dynamic}"                          # 输出目录

# Balanced plan: dynamic latent size [32, 256] with 50% compression ratio
LATENT_SIZE="${LATENT_SIZE:-256}"           # Maximum latent token count
MIN_LATENT_SIZE="${MIN_LATENT_SIZE:-32}"     # Minimum latent token count
LATENT_SCALE_FACTOR="${LATENT_SCALE_FACTOR:-0.5}"  # 50% compression ratio (balanced plan)
ALIGN_WEIGHT="${ALIGN_WEIGHT:-1.0}"        # 对齐损失权重（设 0 可只训练 CE）
EPOCHS="${EPOCHS:-1}"                      # 轮数
BSZ="${BSZ:-1}"                            # 每卡 batch size
GRAD_ACCUM="${GRAD_ACCUM:-8}"              # 梯度累积步数
MAX_SEQ_LEN="${MAX_SEQ_LEN:-3072}"         # 最长序列长度（图片resize后可增大）

# Memory and NCCL configuration
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}" # 内存碎片控制
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"                      # NCCL调试级别
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"  # 异步错误处理（使用新变量名）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"  # 增加超时时间到30分钟
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"  # 分布式调试

FSDP="${FSDP:-}"                         # FSDP config (leave empty, use DeepSpeed instead)
DEEPSPEED="${DEEPSPEED:-./ds_zero3.json}" # DeepSpeed config (compatible with dynamic latent)
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"      # 使用显卡数量
MASTER_PORT="${MASTER_PORT:-29501}"        # torchrun 通信端口（避免29500冲突）
PY_CMD="${PY_CMD:-python}"                 # Python 解释器，可改成 uv run 等

common_args=(
  --stage stage2
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --train_jsonl "$TRAIN_JSONL"
  --teacher_latent_dir "$TEACHER_LATENT_DIR"
  --output_dir "$OUT_DIR"
  --latent_size "$LATENT_SIZE"
  --min_latent_size "$MIN_LATENT_SIZE"
  --latent_scale_factor "$LATENT_SCALE_FACTOR"
  --align_weight "$ALIGN_WEIGHT"
  --max_seq_len "$MAX_SEQ_LEN"
  --epochs "$EPOCHS" --bsz "$BSZ" --grad_accum_steps "$GRAD_ACCUM"
)

if [[ -n "$DEEPSPEED" ]]; then
  common_args+=(--deepspeed "$DEEPSPEED")
fi

if [[ -n "$FSDP" ]]; then
  common_args+=(--fsdp "$FSDP")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  uv run torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m SFT.main "${common_args[@]}"
else
  uv run -m SFT.main "${common_args[@]}"
fi

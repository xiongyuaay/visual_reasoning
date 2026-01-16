## 目标
训练一个多模态模型在输出时满足格式：

`<abs_vis_token>...</abs_vis_token>\n{final_answer}`

其中 `...` 代表**隐式思维链**：训练时用连续向量对齐“思维链图片”，推理时不再需要思维链图片。

### 关键约束（训练阶段）
- `question_image_paths`（用户输入图）放在 **user** turn，所有 token 都能看见
- `thought_image_paths`（思维链图，可多张）放在 **assistant** 开头；只有 `<abs_vis_token_pad>`（latent 位置）能看见，答案 token 看不见

---

## 数据格式（JSONL）
每行一个样本，字段：
- `question`: string
- `answer`: string（GSM8K 风格，脚本会提取 `####` 后的 final answer）
- `question_image_paths`: list[string]（可为空，顺序有意义）
- `thought_image_paths`: list[string]（可为空；stage2/teacher 预计算需要非空，顺序有意义）

---

## 目录结构
- `SFT/main.py`：训练入口（stage1 / stage2 / stage3）
- `SFT/precompute_teacher_latents.py`：离线预计算 teacher latents（来自 thought images）
- `SFT/collate.py`：collator（拼 chat template + 多图输入）
- `SFT/attn_mask.py`：4D attention mask（只让 latent 看到 thought images）
- `SFT/trainer.py`：latent 对齐 trainer（CE(answer) + align）
- `SFT/patch_qwen25vl_latent.py` + `SFT/patches/qwen2_5_vl_latent.py`：patched Qwen2.5-VL（支持 `attention_mask_4d` / `latent_mode`）

---

## 训练流程
### Stage1（格式 warmup，不用图）
让模型熟悉输出结构：先输出 `<abs_vis_token>...</abs_vis_token>`，再输出答案。

### Teacher latents 预计算（离线）
对每个样本，用 `thought_image_paths` 作为 assistant 开头的图像输入，抽取 latent 段的连续向量并保存到：
`teacher_latent_dir/latent_last_layer_jsonl_{idx:06d}.pt`

### Stage2（有 thought images + 可见性约束）
输入包含 question images + thought images，但用 4D mask 保证：
- latent 可以看 thought images
- answer 看不到 thought images，只能依赖 question + question images + latent

### Stage3（无 thought images，推理阶段等价）
训练时不再输入 thought images，只输入 question + question images；latent 段长度 K 在训练时随机采样（1..Kmax），并用离线 teacher latents 监督对齐。

---

## 如何运行
### 安装依赖

```bash
cd /netdisk/xiongyuan/VisualReasoning
pip install -r SFT/requirements.txt
```

### Stage1

```bash
python -m SFT.main \
  --stage stage1 \
  --model_name_or_path /netcache/huggingface/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /path/to/train.jsonl \
  --output_dir /path/to/out_stage1 \
  --latent_size 32 \
  --epochs 1 --bsz 1 --grad_accum_steps 8
```

### 预计算 teacher latents

```bash
python -m SFT.precompute_teacher_latents \
  --model_name_or_path /path/to/out_stage1 \
  --train_jsonl /path/to/train.jsonl \
  --output_dir /path/to/teacher_latents \
  --k_max 32
```

### Stage2

```bash
python -m SFT.main \
  --stage stage2 \
  --model_name_or_path /path/to/out_stage1 \
  --train_jsonl /path/to/train.jsonl \
  --teacher_latent_dir /path/to/teacher_latents \
  --output_dir /path/to/out_stage2 \
  --latent_size 32 \
  --align_weight 1.0 \
  --epochs 1 --bsz 1 --grad_accum_steps 8
```

### Stage3

```bash
python -m SFT.main \
  --stage stage3 \
  --model_name_or_path /path/to/out_stage2 \
  --train_jsonl /path/to/train.jsonl \
  --teacher_latent_dir /path/to/teacher_latents \
  --output_dir /path/to/out_stage3 \
  --latent_size 32 \
  --align_weight 1.0 \
  --epochs 1 --bsz 1 --grad_accum_steps 8
```

---

## 代码卫生
建议做一次关键字扫描时跳过 checkpoint/vocab 文件（例如 `SFT/ckpt_stage1/`），因为 tokenizer 词表里可能自然包含某些英文子词，容易误报。
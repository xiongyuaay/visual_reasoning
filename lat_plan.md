# Latent-mode SFT (no-monet)

## Goal

- Implement **continuous latent thought** generation (latent vectors) using a patched `latent_mode` forward.
- Support **two image types**, each can be **one or many**:
  - **question_images**: appear in the **user** turn; visible to all tokens (normal VL behavior).
  - **thought_images**: appear at the **start of the assistant** turn; visibility is restricted.
- Training stages:
  - **Stage1**: warm up the output format: `<abs_vis_token> ... </abs_vis_token>` then answer (no images required).
  - **Stage2**: input includes question + question_images AND thought_images, but enforce:
    - latent positions can attend to thought-image tokens
    - all other positions (including answer tokens) cannot attend to thought-image tokens
    - everyone can still attend to question-image tokens
  - **Stage3**: **no thought_images input**; model must answer from question + question_images by generating **variable-length** latent segment and then the final answer, supervised by offline teacher latents.
- Constraint: under `SFT/`, **do not include substring `monet`** anywhere in new/edited code/docs; do not import from `/Monet`.

## Data/IO contract (JSONL fields)

- Training JSONL provides per sample:
  - `question: str`
  - `answer: str` (GSM8K style; we extract `answer_final`)
  - `question_image_paths: List[str]` (0..N, order matters; included in user content in order)
  - `thought_image_paths: List[str]` (0..N, order matters; included at assistant start in order)

## High-level design

```mermaid
flowchart TD
  data[DatasetRow idx,question,qImgs?,tImgs?,answer] --> s1[Stage1Collator]
  data --> pre[PrecomputeTeacherLatents]
  data --> s2[Stage2Collator]
  data --> s3[Stage3Collator]

  s1 --> t1[TRL SFTTrainer]

  pre --> latFiles[latent_{layer}_{dataset}_{idx}.pt]

  s2 --> t2[LatentTrainerStage2]
  latFiles --> t2

  s3 --> t3[LatentTrainerStage3]
  latFiles --> t3

  patch[VendoredQwen25VLLatentPatch] --> t2
  patch --> t3
```

## Implementation steps

### 1) Vendor a Qwen2.5-VL latent patch inside `SFT/` (no external deps)

- Add a self-contained patch module that replaces `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl` at import time.
- Required behaviors:
  - `latent_mode=True`: autoregressively produce continuous latent vectors at `<abs_vis_token_pad>` positions; return `(ce_patch_pos, ce_patch_vec)`.
  - `latent_mode=False`: inject `ce_patch_vec` into `inputs_embeds` at `ce_patch_pos` and compute CE.
  - alignment loss: consume `teacher_hidden_states_for_alignment` (teacher latent sequence) and `alignment_poss` (student latent positions) and produce `alignment_loss`.

Files:

- `SFT/patch_qwen25vl_latent.py`
- `SFT/patches/qwen2_5_vl_latent.py`

### 2) Update dataset loader to read multi-image paths from JSONL

- Update `SFT/schemas.py` and `SFT/data.py`:
  - parse `question_image_paths: List[str]` (default to empty list)
  - parse `thought_image_paths: List[str]` (default to empty list)
  - keep `idx` from enumeration order

Files:

- `SFT/schemas.py`, `SFT/data.py`

### 3) Offline precompute: teacher latents from thought_images

- Add `SFT/precompute_teacher_latents.py`:
  - applies the latent patch
  - builds a conversation where:
    - user: `[question_image_1..N, text(question)]`
    - assistant: `[thought_image_1..M, text("<abs_vis_token>" + <abs_vis_token_pad>*K_max + "</abs_vis_token>")]`
  - runs `latent_mode=True` with `output_latent_embeds=True` at positions corresponding to the latent span
  - saves per-sample:
    - `latent_{alignment_layer}_{dataset_name}_{idx:06d}.pt`
    - `latent: Tensor[num_layers, K_max, hidden_dim]`

Notes:

- If `thought_image_paths` is empty, skip precompute for that sample.

Files:

- `SFT/precompute_teacher_latents.py`

### 4) Build 4D attention mask (Stage2 thought-image visibility rule)

- Implement `SFT/attn_mask.py` that builds an attention mask enforcing:
  - standard causal mask
  - question-image tokens remain visible as normal (no additional blocking)
  - ALL thought-image tokens (assistant-start images) are visible **only** to latent positions (`<abs_vis_token_pad>` span; optionally include wrapper)
  - answer tokens cannot attend to thought-image tokens but can attend to question + question-images + latents

Files:

- `SFT/attn_mask.py`

### 5) Stage1: format warmup (no images)

- Stage1 target:
  - assistant: `<abs_vis_token><abs_vis_token_pad>*K</abs_vis_token>\n{final_answer}`
- Labels:
  - supervise answer tokens only (mask wrappers and all `<abs_vis_token_pad>` tokens)

Files:

- `SFT/main.py` (stage wiring)
- `SFT/collate.py` (new collators)

### 6) Stage2: latent-mode training with both image types present

- Conversation format:
  - user: `[question_image_1..N, text(question)]`
  - assistant target: `[thought_image_1..M, text("<abs_vis_token>" + <abs_vis_token_pad>*K_i + "</abs_vis_token>\n{answer}")]`
- Training:
  - load teacher latent sequence from disk (precomputed K_max)
  - choose `K_i`, downsample teacher latents to `K_i`
  - run `latent_mode=True` with `attention_mask_4d` to produce student latents and alignment loss
  - run CE forward injecting latents; CE supervises answer tokens only

Files:

- `SFT/trainer.py` (latent-mode stage2 trainer)
- `SFT/collate.py`
- `SFT/main.py`

### 7) Stage3: no thought_images; variable-length latent; answer from question+question_images

- Inputs:
  - user: `[question_image_1..N, text(question)]`
  - assistant target: `<abs_vis_token><abs_vis_token_pad>*K_i</abs_vis_token>\n{answer}`
- Supervision:
  - teacher latents come from precomputed thought-image latents (K_max), downsampled to `K_i`
- Variable K:
  - training samples `K_i` from a configurable distribution
  - inference stops on `</abs_vis_token>` and uses a hard cap `K_max`

Files:

- `SFT/trainer.py`, `SFT/collate.py`, optional `SFT/generation.py`

### 8) Scripts + docs

- Add runnable scripts:
  - `SFT/run_stage1.sh`
  - `SFT/run_precompute_latents.sh`
  - `SFT/run_stage2.sh`
  - `SFT/run_stage3.sh`
- Update `SFT/README.md`:
  - multi-image placement rules
  - pipeline
  - masking semantics
  - grep check ensuring no `monet` substring under `SFT/`

## Validation checklist

- Stage1 runs.
- Precompute saves `latent_*.pt`.
- Stage2: answer tokens cannot attend to thought-image tokens.
- Stage3: no thought-images in inputs.
- Inference stops latent generation on `</abs_vis_token>` and respects `K_max`.
- No `monet` substring under `SFT/`.
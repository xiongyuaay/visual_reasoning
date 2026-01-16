from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import AutoProcessor

# NOTE: `trl` no longer exports `set_seed` in newer releases. Prefer the stable
# implementation from `transformers`, with fallbacks for older environments.

from transformers import set_seed  # type: ignore
from qwen_vl_utils import process_vision_info

from .collate import TokenBundle, add_special_tokens
from .data import load_gsm8k
from .patch_qwen25vl_latent import apply_qwen25vl_latent_patch
from .trainer import _resample_latents


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k_max", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for precompute forward passes.")
    p.add_argument("--num_shards", type=int, default=1, help="Total shards for parallel precompute (multi-process).")
    p.add_argument("--shard_id", type=int, default=0, help="This process shard id in [0, num_shards).")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--alignment_layer", choices=["last_layer"], default="last_layer")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_dataset(args: argparse.Namespace) -> Dataset:
    samples = load_gsm8k(args.train_jsonl, limit=args.max_samples)
    rows: List[Dict[str, Any]] = []
    for s in samples:
        rows.append(
            {
                "idx": s.idx,
                "question": s.question,
                "answer": s.answer,
                "question_image_paths": s.question_image_paths,
                "thought_image_paths": s.thought_image_paths,
            }
        )
    return Dataset.from_list(rows)


def _build_conv(ex: Dict[str, Any]) -> List[Dict]:
    user_content: List[Dict] = [{"type": "image", "image": p} for p in (ex.get("question_image_paths") or [])] + [
        {"type": "text", "text": ex["question"]}
    ]
    # Teacher target variant:
    # - Put thought images at the start of assistant turn (as required by the masking design).
    # - Do NOT add <abs_vis_token_pad> slots on teacher. We'll extract thought-image semantic token
    #   hidden states directly and compress them to a fixed KxH target for student alignment.
    assistant_content: List[Dict] = [{"type": "image", "image": p} for p in (ex.get("thought_image_paths") or [])] + [
        {"type": "text", "text": " "}
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _find_vision_blocks(
    input_ids_1d: torch.Tensor, *, v_start_id: int, v_end_id: int
) -> List[tuple[int, int]]:
    """
    Find contiguous blocks delimited by <|vision_start|> ... <|vision_end|>.
    Returns (start_idx, end_idx) inclusive, in left-to-right order.
    """
    ids = input_ids_1d.tolist()
    blocks: List[tuple[int, int]] = []
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == int(v_start_id):
            j = i + 1
            while j < n and ids[j] != int(v_end_id):
                j += 1
            if j >= n:
                break
            blocks.append((i, j))
            i = j + 1
        else:
            i += 1
    return blocks


def _pad_positions_in_block(
    input_ids_1d: torch.Tensor, *, start: int, end: int, img_pad_id: int
) -> List[int]:
    ids = input_ids_1d.tolist()
    poss: List[int] = []
    for i in range(int(start), int(end) + 1):
        if ids[i] == int(img_pad_id):
            poss.append(i)
    return poss


@torch.inference_mode()
def main() -> None:
    args = get_args()
    set_seed(int(args.seed))
    os.makedirs(args.output_dir, exist_ok=True)

    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be positive")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive")

    apply_qwen25vl_latent_patch()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    toks: TokenBundle = add_special_tokens(processor)

    try:
        from transformers import AutoModelForVision2Seq as _AutoModel  # type: ignore
    except Exception:
        from transformers import AutoModelForCausalLM as _AutoModel  # type: ignore

    model = _AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.eval()

    # Configure latent ids on model config for downstream logic in patched forward
    model.config.latent_token_id = int(toks.abs_pad_id)
    model.config.latent_start_id = int(toks.abs_start_id)
    model.config.latent_end_id = int(toks.abs_end_id)
    model.config.answer_start_pattern = toks.answer_start_pattern.tolist()

    ds = build_dataset(args)

    # Shard indices for parallel precompute (no DDP needed)
    all_rows = [ex for ex in ds if ex.get("thought_image_paths")]
    shard_rows = [ex for i, ex in enumerate(all_rows) if (i % int(args.num_shards)) == int(args.shard_id)]

    pbar = None
    try:
        from tqdm import tqdm  # type: ignore

        pbar = tqdm(total=len(shard_rows), desc="precompute_teacher_latents", unit="samples")
    except Exception:
        pbar = None

    bs = int(args.batch_size)
    for start in range(0, len(shard_rows), bs):
        chunk = shard_rows[start : start + bs]
        if not chunk:
            continue

        convs = [_build_conv(ex) for ex in chunk]
        texts = [processor.apply_chat_template(c, tokenize=False) for c in convs]
        images, _ = process_vision_info(convs)

        # Handle case when there are no images
        if images is None:
            images = []

        expected = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
        if expected != len(images):
            raise ValueError(f"[precompute] batch_start={start}: image slot mismatch: text expects {expected}, got {len(images)}")

        # Pass None to processor when there are no images to avoid empty tensor issues
        batch = processor(
            text=texts,
            images=images if len(images) > 0 else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        )
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward once for the whole chunk (non-latent): extract thought-image semantic token hidden states.
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            loss_type=[],
            latent_mode=False,
            # Patched model forward currently expects `alignment_poss` to be a list; provide empty per-sample lists.
            alignment_poss=[[] for _ in range(len(chunk))],
            output_hidden_states=True,
            return_dict=True,
        )

        if out.hidden_states is None or len(out.hidden_states) != len(chunk):
            raise RuntimeError(
                f"[precompute] batch_start={start}: missing/invalid hidden_states (got {0 if out.hidden_states is None else len(out.hidden_states)})"
            )

        for b, ex in enumerate(chunk):
            idx = int(ex["idx"])
            hs_b = out.hidden_states[b]
            if not torch.is_tensor(hs_b) or hs_b.dim() != 3:
                raise RuntimeError(f"[precompute] idx={idx}: unexpected hidden_states[{b}] shape {getattr(hs_b, 'shape', None)}")
            tok_hidden: torch.Tensor = hs_b[-1]  # [seq_len, H]

            input_ids_1d = batch["input_ids"][b]
            n_q = len(ex.get("question_image_paths") or [])
            n_t = len(ex.get("thought_image_paths") or [])

            blocks = _find_vision_blocks(input_ids_1d, v_start_id=toks.v_start_id, v_end_id=toks.v_end_id)
            if len(blocks) != (n_q + n_t):
                raise ValueError(
                    f"[precompute] idx={idx}: vision block mismatch: found={len(blocks)} expected={n_q + n_t} "
                    f"(q={n_q}, t={n_t})"
                )

            thought_blocks = blocks[n_q : n_q + n_t]
            thought_poss: List[int] = []
            for (s, e) in thought_blocks:
                thought_poss.extend(_pad_positions_in_block(input_ids_1d, start=s, end=e, img_pad_id=toks.img_pad_id))

            if len(thought_poss) == 0:
                raise RuntimeError(
                    f"[precompute] idx={idx}: no thought <|image_pad|> positions found; max_seq_len may be too small."
                )

            raw = tok_hidden[torch.tensor(thought_poss, device=tok_hidden.device, dtype=torch.long), :]  # [L,H]
            teacher = _resample_latents(raw, int(args.k_max))  # [K,H]

            save_path = os.path.join(args.output_dir, f"latent_{args.alignment_layer}_jsonl_{idx:06d}.pt")
            torch.save(
                {
                    "dataset_name": "jsonl",
                    "sample_id": idx,
                    "latent": teacher.detach().cpu(),
                    "latent_source": "thought_image_semantic",
                    "latent_source_num_thought_tokens": int(raw.shape[0]),
                },
                save_path,
            )

        if pbar is not None:
            pbar.update(len(chunk))

    if pbar is not None:
        pbar.close()


if __name__ == "__main__":
    main()


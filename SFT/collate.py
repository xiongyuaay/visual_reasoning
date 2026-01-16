from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import random
import torch
from qwen_vl_utils import process_vision_info

from .attn_mask import SpecialTokenIds, build_attention_mask_4d_thought_hidden, latent_positions_after_assistant
from .prompting import build_assistant_with_latents


@dataclass(frozen=True)
class TokenBundle:
    abs_start_id: int
    abs_end_id: int
    abs_pad_id: int
    img_pad_id: int
    v_start_id: int
    v_end_id: int
    answer_start_pattern: torch.Tensor  # 1D tensor


def add_special_tokens(processor: Any) -> TokenBundle:
    tok = processor.tokenizer
    tok.add_tokens("<abs_vis_token_pad>", special_tokens=True)
    tok.add_tokens("<abs_vis_token>", special_tokens=True)
    tok.add_tokens("</abs_vis_token>", special_tokens=True)

    abs_pad_id = tok("<abs_vis_token_pad>", add_special_tokens=False)["input_ids"][0]
    abs_start_id = tok("<abs_vis_token>", add_special_tokens=False)["input_ids"][0]
    abs_end_id = tok("</abs_vis_token>", add_special_tokens=False)["input_ids"][0]

    v_start_id = tok("<|vision_start|>", add_special_tokens=False)["input_ids"][0]
    v_end_id = tok("<|vision_end|>", add_special_tokens=False)["input_ids"][0]
    img_pad_id = tok("<|image_pad|>", add_special_tokens=False)["input_ids"][0]

    answer_start_pattern = tok("<|im_start|>assistant", return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    return TokenBundle(
        abs_start_id=int(abs_start_id),
        abs_end_id=int(abs_end_id),
        abs_pad_id=int(abs_pad_id),
        img_pad_id=int(img_pad_id),
        v_start_id=int(v_start_id),
        v_end_id=int(v_end_id),
        answer_start_pattern=answer_start_pattern,
    )


def _make_labels_after_assistant(
    input_ids: torch.Tensor,
    *,
    answer_start_pattern: torch.Tensor,
    pad_token_id: int,
    ignore_ids: List[int],
) -> torch.Tensor:
    labels = input_ids.clone()
    B, S = labels.shape

    # Mask everything up to and including the first assistant-start pattern
    for b in range(B):
        row = input_ids[b]
        # naive scan
        start = -1
        pat = answer_start_pattern.to(row.device)
        L, M = int(row.numel()), int(pat.numel())
        for i in range(0, L - M + 1):
            if torch.equal(row[i : i + M], pat):
                start = i + M
                break
        if start < 0:
            labels[b, :] = -100
        else:
            labels[b, :start] = -100

    # Mask pads + ignored token ids globally
    labels[labels == int(pad_token_id)] = -100
    for tid in ignore_ids:
        labels[labels == int(tid)] = -100
    return labels


def _build_conversation_text_only(question: str, assistant_text: str) -> List[Dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def _build_conversation_with_images(
    *,
    question: str,
    question_image_paths: List[str],
    thought_image_paths: List[str],
    assistant_text: str,
) -> List[Dict]:
    user_content: List[Dict] = [{"type": "image", "image": p} for p in question_image_paths] + [
        {"type": "text", "text": question}
    ]
    assistant_content: List[Dict] = [{"type": "image", "image": p} for p in thought_image_paths] + [
        {"type": "text", "text": assistant_text}
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def collate_stage1(
    examples: List[Dict[str, Any]],
    *,
    processor: Any,
    toks: TokenBundle,
    latent_size: int,
    min_latent_size: int = 1,
    variable_latent: bool = False,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    convs = []
    for ex in examples:
        k = int(latent_size)
        if variable_latent:
            k = random.randint(int(min_latent_size), int(latent_size))
        convs.append(
            _build_conversation_text_only(
                ex["question"],
                build_assistant_with_latents(ex["answer"], latent_size=k),
            )
        )
    texts = [processor.apply_chat_template(c, tokenize=False) for c in convs]
    batch = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )

    ignore_ids = [toks.abs_start_id, toks.abs_end_id, toks.abs_pad_id]
    batch["labels"] = _make_labels_after_assistant(
        batch["input_ids"],
        answer_start_pattern=toks.answer_start_pattern,
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_ids=ignore_ids,
    )
    return batch


def collate_stage2_or_3(
    examples: List[Dict[str, Any]],
    *,
    processor: Any,
    toks: TokenBundle,
    latent_size: int,
    min_latent_size: int = 1,
    variable_latent: bool = False,
    max_seq_len: int,
    include_thought_images: bool,
    dynamic_latent_size: bool = False,
    latent_scale_factor: float = 0.25,
    teacher_latent_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    # FSDP compatibility: Compute each sample's desired latent count, then pad to batch max
    # This ensures all samples use the same latent count (same loop iterations in forward)
    desired_latent_counts = []

    for ex in examples:
        k = int(latent_size)

        # Dynamic latent size: compute desired count based on teacher latent source tokens
        if dynamic_latent_size and teacher_latent_dir:
            try:
                import os
                fname = f"latent_last_layer_jsonl_{int(ex['idx']):06d}.pt"
                path = os.path.join(teacher_latent_dir, fname)
                if os.path.isfile(path):
                    latent_data = torch.load(path, map_location="cpu", weights_only=False)
                    orig_tokens = latent_data.get("latent_source_num_thought_tokens", 0)
                    if orig_tokens > 0:
                        # Calculate dynamic latent size
                        k = max(int(min_latent_size), min(int(latent_size), int(orig_tokens * latent_scale_factor)))
            except Exception as e:
                pass

        if variable_latent and not dynamic_latent_size:
            k = random.randint(int(min_latent_size), int(latent_size))

        desired_latent_counts.append(k)

    # Pad to batch maximum for FSDP sync (all samples use same latent count → same forward loop count)
    if desired_latent_counts:
        batch_max_latent = max(desired_latent_counts)
    else:
        batch_max_latent = int(latent_size)

    # CRITICAL: Sync batch_max_latent across all GPUs for FSDP compatibility
    # When per_device_batch_size=1, each GPU processes different samples with different batch_max values
    # We must use the GLOBAL max across all ranks to ensure same forward loop count
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Convert to tensor on CPU first (collate_fn runs before moving to GPU)
            batch_max_tensor = torch.tensor(batch_max_latent, dtype=torch.long, device='cpu')
            # Move to current device for all_reduce
            if torch.cuda.is_available():
                batch_max_tensor = batch_max_tensor.cuda()
            # Synchronize: take the maximum across all ranks
            dist.all_reduce(batch_max_tensor, op=dist.ReduceOp.MAX)
            batch_max_latent = int(batch_max_tensor.item())
    except Exception as e:
        # If distributed not available or errors, use local max (single GPU training)
        pass

    convs = []
    for ex in examples:
        thought_paths = ex["thought_image_paths"] if include_thought_images else []
        convs.append(
            _build_conversation_with_images(
                question=ex["question"],
                question_image_paths=list(ex.get("question_image_paths") or []),
                thought_image_paths=list(thought_paths or []),
                assistant_text=build_assistant_with_latents(ex["answer"], latent_size=batch_max_latent),  # Use batch max
            )
        )

    texts = [processor.apply_chat_template(c, tokenize=False) for c in convs]
    image_inputs, _ = process_vision_info(convs)

    # Handle case when there are no images (stage3 with no question images)
    if image_inputs is None:
        image_inputs = []

    # Each image slot begins with '<|vision_start|><|image_pad|>'
    expected = sum(t.count("<|vision_start|><|image_pad|>") for t in texts)
    if expected != len(image_inputs):
        raise ValueError(f"Image slot mismatch: text expects {expected} images, but got {len(image_inputs)}.")

    # Pass None to processor when there are no images to avoid empty tensor issues
    batch = processor(
        text=texts,
        images=image_inputs if len(image_inputs) > 0 else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )

    # 4D mask: hide assistant-start thought-image tokens from non-latent queries
    token_ids = SpecialTokenIds(
        img_pad=toks.img_pad_id,
        v_start=toks.v_start_id,
        v_end=toks.v_end_id,
        abs_pad=toks.abs_pad_id,
    )
    batch["attention_mask_4d"] = build_attention_mask_4d_thought_hidden(
        input_ids=batch["input_ids"],
        pad_mask=batch["attention_mask"],
        answer_start_pattern=toks.answer_start_pattern.to(batch["input_ids"].device),
        token_ids=token_ids,
    )

    # latent positions (alignment positions) after assistant start
    batch["alignment_poss"] = latent_positions_after_assistant(
        batch["input_ids"],
        toks.answer_start_pattern.to(batch["input_ids"].device),
        toks.abs_pad_id,
    )

    # CRITICAL FSDP FIX: Verify all samples have same latent count
    # If truncation happened, some samples may have fewer latents than batch_max
    actual_latent_counts = [len(p) for p in batch["alignment_poss"]]
    if len(set(actual_latent_counts)) > 1:
        # Mismatch detected! This will cause FSDP collective mismatch
        import warnings
        warnings.warn(
            f"Latent count mismatch after tokenization: {actual_latent_counts}. "
            f"Expected all to be {batch_max_latent}. This may cause FSDP errors. "
            f"Consider increasing max_seq_len or decreasing latent_size."
        )

    # Sync actual max across all ranks (in case truncation differs per GPU)
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            max_actual = max(actual_latent_counts) if actual_latent_counts else 0
            max_actual_tensor = torch.tensor(max_actual, dtype=torch.long, device='cpu')
            if torch.cuda.is_available():
                max_actual_tensor = max_actual_tensor.cuda()
            dist.all_reduce(max_actual_tensor, op=dist.ReduceOp.MAX)
            global_max_actual = int(max_actual_tensor.item())

            # If any rank has different count, this will cause mismatch
            if max_actual != global_max_actual or len(set(actual_latent_counts)) > 1:
                raise RuntimeError(
                    f"FSDP incompatible: Different ranks have different latent counts. "
                    f"Local counts: {actual_latent_counts}, Global max: {global_max_actual}. "
                    f"Increase max_seq_len from {max_seq_len} or reduce latent_size/images."
                )
    except RuntimeError:
        raise
    except Exception:
        pass

    # Labels: supervise answer tokens only (ignore wrappers/latents and vision tokens)
    ignore_ids = [
        toks.abs_start_id,
        toks.abs_end_id,
        toks.abs_pad_id,
        toks.img_pad_id,
        toks.v_start_id,
        toks.v_end_id,
    ]
    batch["labels"] = _make_labels_after_assistant(
        batch["input_ids"],
        answer_start_pattern=toks.answer_start_pattern.to(batch["input_ids"].device),
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_ids=ignore_ids,
    )

    # Carry metadata needed for offline latent lookup
    batch["metadata"] = [{"dataset_name": "jsonl", "sample_id": int(ex["idx"])} for ex in examples]
    batch["latent_sizes"] = torch.tensor(
        [len(p) for p in batch["alignment_poss"]], dtype=torch.long
    )  # Actual latent token count in each sample (should all equal batch_max_latent)
    batch["desired_latent_counts"] = torch.tensor(
        desired_latent_counts, dtype=torch.long
    )  # Original desired counts (before padding to batch max)
    batch["batch_max_latent"] = batch_max_latent  # The batch-level max used for padding
    return batch



from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SpecialTokenIds:
    img_pad: int
    v_start: int
    v_end: int
    abs_pad: int


def _find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:
    """Return the first start index of `pattern` in `row`, else -1."""
    assert row.dim() == 1 and pattern.dim() == 1
    L = int(row.numel())
    M = int(pattern.numel())
    if M == 0 or L < M:
        return -1
    # naive scan; sequences are short enough for training-time use
    for i in range(0, L - M + 1):
        if torch.equal(row[i : i + M], pattern):
            return i
    return -1


def assistant_start_positions(input_ids: torch.Tensor, answer_start_pattern: torch.Tensor) -> List[int]:
    """
    Find the end position (exclusive) of the first '<|im_start|>assistant' pattern per sample.
    Returns a python list of ints of length B.
    """
    poss: List[int] = []
    for b in range(int(input_ids.size(0))):
        row = input_ids[b]
        s = _find_subsequence(row, answer_start_pattern)
        poss.append(0 if s < 0 else int(s + answer_start_pattern.numel()))
    return poss


def latent_positions_after_assistant(
    input_ids: torch.Tensor,
    answer_start_pattern: torch.Tensor,
    abs_pad_id: int,
) -> List[List[int]]:
    """
    Return positions of abs_pad tokens AFTER the first assistant-start pattern per sample.
    """
    B, S = input_ids.shape
    ans_ends = assistant_start_positions(input_ids, answer_start_pattern)
    out: List[List[int]] = []
    for b in range(B):
        pos = (input_ids[b] == int(abs_pad_id)).nonzero(as_tuple=False).flatten().tolist()
        out.append([int(p) for p in pos if int(p) >= ans_ends[b]])
    return out


def build_attention_mask_4d_thought_hidden(
    *,
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    answer_start_pattern: torch.Tensor,
    token_ids: SpecialTokenIds,
) -> Dict[str, torch.Tensor]:
    """
    Build a boolean 4D attention mask [B, 1, S, S] for Qwen2.5-VL patched model.\n
    Rules:\n
    - causal attention for all tokens\n
    - question-image tokens (before assistant start) are visible as normal\n
    - thought-image tokens (assistant-start images) are visible ONLY to abs_pad (latent) query positions\n
    """
    assert input_ids.dim() == 2 and pad_mask.dim() == 2
    B, S = input_ids.shape
    device = input_ids.device
    qk = torch.tril(torch.ones((S, S), device=device, dtype=torch.bool))  # [S,S]
    mask = qk.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S).clone()  # [B,1,S,S]

    q_valid = pad_mask.to(torch.bool)
    k_valid = pad_mask.to(torch.bool)
    mask &= q_valid[:, None, :, None]
    mask &= k_valid[:, None, None, :]

    # Identify assistant boundary
    ans_ends = assistant_start_positions(input_ids, answer_start_pattern)  # list[int]
    idx = torch.arange(S, device=device)[None, :].expand(B, S)

    # Thought-image keys are image tokens that appear after the assistant start.
    is_img_token = input_ids.eq(int(token_ids.img_pad))
    is_thought_img_key = is_img_token & (idx >= torch.tensor(ans_ends, device=device)[:, None])

    # Latent queries are abs_pad tokens after assistant start.
    is_latent_query = input_ids.eq(int(token_ids.abs_pad))
    is_non_latent_query = ~is_latent_query

    # Block: non-latent queries cannot attend to thought-image keys.
    # Implement by setting mask[b,0,q,k]=False where q is non-latent and k is thought-img.
    for b in range(B):
        if not bool(is_thought_img_key[b].any()):
            continue
        q_idx = is_non_latent_query[b].nonzero(as_tuple=False).flatten()
        k_idx = is_thought_img_key[b].nonzero(as_tuple=False).flatten()
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            continue
        mask[b, 0].index_put_((q_idx[:, None], k_idx[None, :]), torch.zeros((q_idx.numel(), k_idx.numel()), device=device, dtype=torch.bool))

    return {"full_attention": mask}



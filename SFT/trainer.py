from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from trl import SFTTrainer


def _load_teacher_latents(
    teacher_latent_dir: str,
    *,
    dataset_name: str,
    sample_id: int,
    alignment_layer: str = "last_layer",
) -> torch.Tensor:
    """
    Load offline teacher latents from:
      latent_{alignment_layer}_{dataset_name}_{sample_id:06d}.pt

    Expected tensor under key 'latent':
      - [K, H] for last-layer latents, or
      - [L, K, H] for all-layer latents.
    """
    fname = f"latent_{alignment_layer}_{dataset_name}_{int(sample_id):06d}.pt"
    path = os.path.join(teacher_latent_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing teacher latent file: {path}")
    obj = torch.load(path, map_location="cpu")
    t = obj.get("latent")
    if not torch.is_tensor(t):
        raise ValueError(f"Invalid latent in {path}: expected Tensor, got {type(t)}")
    if t.dim() not in (2, 3):
        raise ValueError(f"Invalid latent shape in {path}: expected dim 2/3, got {tuple(t.shape)}")
    return t


def _resample_latents(latents: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resample along the latent length axis to `target_len`.\n
    latents: [K,H] or [L,K,H]\n
    returns: same rank with K replaced by target_len.\n
    """
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if latents.dim() == 2:
        K, H = latents.shape
        if K == target_len:
            return latents
        x = latents.transpose(0, 1).unsqueeze(0)  # [1,H,K]
        y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return y.squeeze(0).transpose(0, 1).contiguous()  # [target_len,H]
    else:
        L, K, H = latents.shape
        if K == target_len:
            return latents
        x = latents.permute(0, 2, 1).contiguous()  # [L,H,K]
        y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)  # [L,H,target]
        return y.permute(0, 2, 1).contiguous()  # [L,target,H]


class CustomTrainerLatentAlign(SFTTrainer):
    """
    Latent alignment training (Stage2/Stage3).\n
    - Uses a patched Qwen2.5-VL model that supports `latent_mode` and `attention_mask_4d`.\n
    - Offline teacher latents are loaded from disk.\n
    - Loss = CE(answer) + align_weight * alignment_loss.\n
    """

    def __init__(
        self,
        *args,
        teacher_latent_dir: str,
        align_weight: float = 1.0,
        alignment_layer: str = "last_layer",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_latent_dir = teacher_latent_dir
        self.align_weight = float(align_weight)
        self.alignment_layer = alignment_layer

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        # inputs expected from collator:
        # - input_ids, attention_mask, labels
        # - pixel_values, image_grid_thw (if images exist)
        # - attention_mask_4d: {"full_attention": BoolTensor[B,1,S,S]}
        # - alignment_poss: List[List[int]] positions of abs_pad tokens after assistant start
        # - metadata: List[{"dataset_name": str, "sample_id": int}]
        metadata: List[Dict[str, Any]] = inputs["metadata"]

        do_alignment = self.align_weight > 0

        # Load + resample teacher latents to match each sample's latent length (only if aligning)
        teacher_latents: List[torch.Tensor] = []
        if do_alignment:
            for b, md in enumerate(metadata):
                t = _load_teacher_latents(
                    self.teacher_latent_dir,
                    dataset_name=str(md["dataset_name"]),
                    sample_id=int(md["sample_id"]),
                    alignment_layer=self.alignment_layer,
                )
                target_len = len(inputs["alignment_poss"][b])
                if target_len <= 0:
                    raise ValueError("alignment_poss is empty; cannot align without latent pads in the sequence.")
                teacher_latents.append(_resample_latents(t, target_len))

        # Move to device/dtype
        device = inputs["input_ids"].device
        dtype = next(model.parameters()).dtype
        teacher_latents = [t.to(device=device, dtype=dtype) for t in teacher_latents] if teacher_latents else None

        # Single forward pass with latent_mode=True
        # The model will compute both latent embeddings and losses in one pass
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            attention_mask_4d=inputs.get("attention_mask_4d"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            labels=inputs["labels"],
            latent_mode=True,
            alignment_poss=inputs["alignment_poss"],
            teacher_hidden_states_for_alignment=teacher_latents,
            loss_type=["ce", "alignment"] if do_alignment else ["ce"],
            output_hidden_states=False,
            return_dict=True,
        )

        ce_loss = out.loss
        alignment_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        loss_dict = getattr(out, "loss_dict", None)
        if isinstance(loss_dict, dict) and "alignment" in loss_dict and loss_dict["alignment"] is not None:
            alignment_loss = loss_dict["alignment"]

        loss = ce_loss + self.align_weight * alignment_loss
        if return_outputs:
            return loss, {"ce_loss": ce_loss.detach(), "alignment_loss": alignment_loss.detach()}
        return loss


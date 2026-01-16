from __future__ import annotations

import importlib.util
import pathlib
import sys


def apply_qwen25vl_latent_patch() -> None:
    """
    Replace `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl` with a local patched module
    that supports:
    - attention_mask_4d
    - latent_mode (collect latent vectors at <abs_vis_token_pad> positions)
    - ce_patch_pos / ce_patch_vec injection
    """
    patch_path = pathlib.Path(__file__).with_name("patches") / "qwen2_5_vl_latent.py"
    if not patch_path.is_file():
        raise FileNotFoundError(f"Patched modeling file not found: {patch_path}")

    spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        str(patch_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {patch_path}")

    patched_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched_mod)  # type: ignore[union-attr]
    sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"] = patched_mod



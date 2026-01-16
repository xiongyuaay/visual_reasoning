from __future__ import annotations

import re
from typing import Dict, List

ABS_START = "<abs_vis_token>"
ABS_END = "</abs_vis_token>"
ABS_PAD = "<abs_vis_token_pad>"

QWEN_IMAGE_PAD = "<|vision_start|><|image_pad|><|vision_end|>"


_ABS_BLOCK_RE = re.compile(
    re.escape(ABS_START) + r".*?" + re.escape(ABS_END),
    flags=re.DOTALL,
)


def build_conversation(question: str, assistant_text: str, system: str = "You are a helpful assistant.") -> List[Dict]:
    """
    Build a Qwen-style conversation list for AutoProcessor.apply_chat_template().
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def assistant_with_thought_image(answer: str) -> str:
    """
    Stage1 target: use a real image placeholder (Qwen-style) as the "thinking process".
    """
    return f"{QWEN_IMAGE_PAD}\n{answer}"


def build_assistant_with_latents(answer: str, latent_size: int) -> str:
    """
    Output format target (per SFT/README.md):
      [<abs_vis_token> ... </abs_vis_token>] | [answer]

    We implement "..." as repeated ABS_PAD tokens.
    """
    latent_tokens = ABS_PAD * max(1, int(latent_size))
    return f"{ABS_START}{latent_tokens}{ABS_END}\n{answer}"


def replace_abs_with_image_pad(text: str) -> str:
    """
    Replace any <abs_vis_token> ... </abs_vis_token> block with a single Qwen image placeholder.
    Used for teacher forward when aligning student latents to thought-image representations.
    """
    return _ABS_BLOCK_RE.sub(QWEN_IMAGE_PAD, text)



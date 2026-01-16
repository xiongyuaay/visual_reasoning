from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GSM8KSample:
    """A single example with optional question images and thought images."""

    idx: int
    question: str
    answer: str
    question_image_paths: List[str]
    thought_image_paths: List[str]


@dataclass
class SFTBatch:
    """Batch dict-like container (used only for type hints / readability)."""

    input_ids: object
    attention_mask: object
    labels: object
    pixel_values: Optional[object] = None
    image_grid_thw: Optional[object] = None



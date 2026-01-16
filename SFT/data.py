from __future__ import annotations

import json
from typing import List, Sequence

from .schemas import GSM8KSample


def load_gsm8k(
    jsonl_path: str,
    limit: int = -1,
) -> List[GSM8KSample]:
    """
    Load samples from JSONL. Each row may include:
      - question_image_paths: [..]  (optional)
      - thought_image_paths: [..]   (optional)
    """
    samples: List[GSM8KSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit != -1 and len(samples) >= limit:
                break
            obj = json.loads(line)
            q = obj.get("question", "")
            # New format only: `answer` is the final answer string (after ####).
            a = (obj.get("answer", "") or "").strip()
            q_imgs: Sequence[str] = obj.get("question_image_paths", []) or []
            t_imgs: Sequence[str] = obj.get("thought_image_paths", []) or []
            samples.append(
                GSM8KSample(
                    idx=idx,
                    question=q,
                    answer=a,
                    question_image_paths=list(q_imgs),
                    thought_image_paths=list(t_imgs),
                )
            )
    return samples



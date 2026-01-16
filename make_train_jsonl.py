from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Any, Dict, List


_FINAL_ANS_RE = re.compile(r"####\s*([^\n]+)")


def _extract_final_answer(gsm8k_answer: str) -> str:
    """
    GSM8K `answer` usually ends with:
      "#### 72"
    We extract the token after #### as the final answer string.
    Fallback: last non-empty line.
    """
    if not gsm8k_answer:
        return ""
    m = _FINAL_ANS_RE.search(gsm8k_answer)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in gsm8k_answer.splitlines() if ln.strip()]
    return lines[-1] if lines else gsm8k_answer.strip()


def _sorted_pngs(dir_path: str, idx: int, pattern: str) -> List[str]:
    """
    Return sorted absolute paths matching pattern with {idx} substituted.
    pattern examples:
      - "{idx:06d}.png"
      - "{idx:06d}_*.png"
      - "{idx:06d}*.png"
    """
    if not dir_path:
        return []
    fp = os.path.join(dir_path, pattern.format(idx=idx))
    files = sorted(glob(fp))
    return [os.path.abspath(p) for p in files if os.path.isfile(p)]


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src_jsonl", required=True, help="Source GSM8K-like jsonl with fields {question, answer}.")
    p.add_argument("--out_jsonl", required=True, help="Output jsonl for SFT.")

    p.add_argument("--question_image_dir", default="", help="Optional dir for question images.")
    p.add_argument(
        "--question_image_pattern",
        default="{idx:06d}_*.png",
        help="Glob pattern inside question_image_dir (python format with {idx}).",
    )

    p.add_argument("--thought_image_dir", default="", help="Optional dir for thought images.")
    p.add_argument(
        "--thought_image_pattern",
        default="{idx:06d}*.png",
        help="Glob pattern inside thought_image_dir (python format with {idx}).",
    )

    p.add_argument("--limit", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = get_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)

    # Progress behavior (hard-coded):
    # - If tqdm is installed, show a progress bar.
    # - Otherwise, print every LOG_EVERY written samples.
    LOG_EVERY = 50

    n_in = 0
    n_out = 0
    with open(args.src_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        iterator = enumerate(fin)
        has_tqdm = False
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(iterator, desc="make_train_jsonl", unit="lines")
            has_tqdm = True
        except Exception:
            # tqdm not installed (or import failed) -> silent fallback to LOG_EVERY prints
            has_tqdm = False

        for idx, line in iterator:
            if args.limit != -1 and n_out >= int(args.limit):
                break
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)
            n_in += 1

            question = obj.get("question", "")
            answer = _extract_final_answer(obj.get("answer", ""))

            q_imgs = _sorted_pngs(args.question_image_dir, idx, args.question_image_pattern)
            t_imgs = _sorted_pngs(args.thought_image_dir, idx, args.thought_image_pattern)

            out = {
                "question": question,
                "answer": answer,
                "question_image_paths": q_imgs,
                "thought_image_paths": t_imgs,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            if (not has_tqdm) and (LOG_EVERY > 0) and (n_out % int(LOG_EVERY) == 0):
                print(f"[make_train_jsonl] wrote={n_out} (last_idx={idx})", flush=True)

    print(f"[make_train_jsonl] read={n_in} wrote={n_out} out={args.out_jsonl}")


if __name__ == "__main__":
    main()


"""
python -m SFT.make_train_jsonl \
  --src_jsonl /netdisk/xiongyuan/data/gsm8k/train.jsonl \
  --out_jsonl /netdisk/xiongyuan/VisualReasoning/data/train/stage1/gsm8k.jsonl \
  --thought_image_dir /netdisk/xiongyuan/VisualReasoning/data/gsm8k/train \
  --thought_image_pattern '{idx:06d}*.png' \
  --question_image_dir /path/to/question_images \
  --question_image_pattern '{idx:06d}_*.png'
"""
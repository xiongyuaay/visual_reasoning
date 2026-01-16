#!/usr/bin/env python3
"""
Verify that the trained model generates latent tokens during inference.

Usage:
    uv run python verify_latent_output.py --model_path <path_to_stage3_model>
"""
from __future__ import annotations

import argparse
import re
from typing import Dict, List

import torch
from transformers import AutoProcessor


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify latent token generation in model inference")
    p.add_argument("--model_path", required=True, help="Path to trained model (e.g., SFT/out_stage3_dynamic)")
    p.add_argument("--max_new_tokens", type=int, default=512, help="Maximum tokens to generate")
    p.add_argument("--num_samples", type=int, default=5, help="Number of test samples to run")
    return p.parse_args()


def build_test_questions() -> List[str]:
    """Build a few test math questions."""
    return [
        "What is 25 + 37?",
        "If a train travels 60 miles per hour for 2.5 hours, how far does it go?",
        "What is 15% of 200?",
        "A rectangle has length 8 cm and width 5 cm. What is its area?",
        "If 3 apples cost $2.40, how much does one apple cost?",
    ]


def count_latent_tokens(text: str) -> Dict[str, int]:
    """
    Count latent-related tokens in generated text.

    Returns dict with:
        - has_latent_block: whether <abs_vis_token>...</abs_vis_token> exists
        - num_latent_pads: number of <abs_vis_token_pad> tokens
        - latent_start_count: number of <abs_vis_token>
        - latent_end_count: number of </abs_vis_token>
    """
    has_start = "<abs_vis_token>" in text
    has_end = "</abs_vis_token>" in text

    # Count padding tokens
    num_pads = text.count("<abs_vis_token_pad>")

    # Count start/end markers (excluding the closing tag)
    start_count = len(re.findall(r"<abs_vis_token>(?!<)", text))
    end_count = text.count("</abs_vis_token>")

    return {
        "has_latent_block": has_start and has_end,
        "num_latent_pads": num_pads,
        "latent_start_count": start_count,
        "latent_end_count": end_count,
    }


def extract_latent_and_answer(text: str) -> tuple[str | None, str | None]:
    """
    Extract latent block and answer from generated text.

    Expected format: <abs_vis_token>...(pads)...</abs_vis_token>\n{answer}
    """
    # Match latent block
    latent_pattern = r"<abs_vis_token>(.*?)</abs_vis_token>"
    latent_match = re.search(latent_pattern, text, re.DOTALL)

    if not latent_match:
        return None, text.strip()

    latent_block = latent_match.group(0)
    # Extract answer (everything after the latent block)
    answer = text[latent_match.end():].strip()

    return latent_block, answer


def main() -> None:
    args = get_args()

    print(f"Loading model from: {args.model_path}")
    print("=" * 80)

    # Apply patch before loading model
    from SFT.patch_qwen25vl_latent import apply_qwen25vl_latent_patch
    apply_qwen25vl_latent_patch()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Load model
    try:
        from transformers import AutoModelForVision2Seq as _AutoModel
    except Exception:
        from transformers import AutoModelForCausalLM as _AutoModel

    model = _AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded on: {device}")
    print(f"Model type: {type(model).__name__}")
    print("=" * 80)

    # Test questions
    questions = build_test_questions()[:args.num_samples]

    total_samples = 0
    successful_samples = 0

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(questions)}")
        print(f"Question: {question}")
        print("-" * 80)

        # Build conversation
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode only the generated part (skip input)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

        print(f"Generated text (raw):\n{repr(generated_text)}")
        print("-" * 80)

        # Analyze latent tokens
        stats = count_latent_tokens(generated_text)
        latent_block, answer = extract_latent_and_answer(generated_text)

        print(f"Analysis:")
        print(f"  - Has latent block: {stats['has_latent_block']}")
        print(f"  - Latent start markers: {stats['latent_start_count']}")
        print(f"  - Latent end markers: {stats['latent_end_count']}")
        print(f"  - Latent pad tokens: {stats['num_latent_pads']}")

        if latent_block:
            print(f"  - Latent block (truncated): {latent_block[:100]}...")
            print(f"  - Answer: {answer[:200]}")
        else:
            print(f"  - No latent block found!")
            print(f"  - Raw output: {generated_text[:200]}")

        total_samples += 1
        if stats['has_latent_block'] and stats['num_latent_pads'] > 0:
            successful_samples += 1
            print(f"  ✓ SUCCESS: Model generated {stats['num_latent_pads']} latent tokens")
        else:
            print(f"  ✗ FAILED: Model did not generate expected latent format")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total samples: {total_samples}")
    print(f"  Successful: {successful_samples}")
    print(f"  Failed: {total_samples - successful_samples}")
    print(f"  Success rate: {successful_samples/total_samples*100:.1f}%")
    print("=" * 80)

    if successful_samples == total_samples:
        print("\n✓ All tests passed! Model correctly generates latent tokens.")
    elif successful_samples > 0:
        print(f"\n⚠ Partial success: {successful_samples}/{total_samples} samples generated latent tokens.")
    else:
        print("\n✗ All tests failed! Model may not have learned latent token generation.")
        print("  Possible issues:")
        print("  - Model checkpoint may be from stage1 (before latent training)")
        print("  - Special tokens may not be properly loaded")
        print("  - Generation parameters may need adjustment")


if __name__ == "__main__":
    main()

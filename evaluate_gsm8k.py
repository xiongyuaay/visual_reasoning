#!/usr/bin/env python3
"""
Evaluate models on GSM8K test set.
Supports both original Qwen2.5-VL and SFT-trained models with latent mode.
"""
from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor

# Use AutoModel for automatic model class selection (standard approach)
try:
    from transformers import AutoModelForVision2Seq as AutoModel
except ImportError:
    from transformers import AutoModelForCausalLM as AutoModel


def extract_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from generated text.
    GSM8K answers are typically in the format "#### 123" or just a number.
    """
    # Try to find #### format first
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')

    # Try to find the last number in the text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove commas and whitespace
    answer = answer.replace(',', '').strip()
    # Convert to float and back to remove trailing zeros
    try:
        return str(float(answer))
    except ValueError:
        return answer


def load_model_and_processor(
    model_path: str,
    use_latent_mode: bool = False,
    device: str = "auto",
    torch_dtype=torch.bfloat16,
):
    """
    Load model and processor.

    Args:
        model_path: Path to model checkpoint
        use_latent_mode: Whether to apply latent mode patch (for SFT models)
        device: Device to load model on. Options:
                - "auto": automatically select device (recommended)
                - "cuda": use default GPU (GPU 0)
                - "cuda:0", "cuda:1", etc.: use specific GPU
                - "cpu": use CPU
        torch_dtype: Data type for model
    """
    # CRITICAL: Apply patch BEFORE loading model!
    # SFT models were saved with patched structure, so we must apply patch first
    if use_latent_mode:
        from SFT.patch_qwen25vl_latent import apply_qwen25vl_latent_patch
        apply_qwen25vl_latent_patch()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,  # Required for local paths
    )

    # Determine device_map
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    # Load model using AutoModel (automatically selects correct class based on config)
    # This works for both original Qwen2.5-VL and SFT-trained models
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,  # Required for local paths
    )
    model.eval()

    return model, processor


def build_inference_prompt(
    question: str,
    question_image_paths: List[str],
    use_latent_mode: bool = False,
    latent_size: Optional[int] = None,
    force_prefix: bool = False,
) -> List[Dict]:
    """
    Build conversation for inference.

    Args:
        question: Question text
        question_image_paths: Paths to question images
        use_latent_mode: Whether model uses latent mode (for SFT models)
        latent_size: Number of latent tokens (only used if force_prefix=True)
        force_prefix: If True, pre-fill latent tokens in prompt (not recommended).
                      If False (default), let model generate latent tokens autoregressively.
    """
    from SFT.prompting import ABS_START, ABS_END, ABS_PAD

    # Build user content
    user_content = []

    # Add question images
    for img_path in question_image_paths:
        user_content.append({"type": "image", "image": img_path})

    # Add question text
    user_content.append({"type": "text", "text": question})

    # Build conversation
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
    ]

    # DEPRECATED: Pre-fill latent tokens (not recommended)
    # Recommended: Let model generate latent tokens autoregressively
    if use_latent_mode and force_prefix and latent_size is not None:
        latent_tokens = ABS_PAD * latent_size
        assistant_prefix = f"{ABS_START}{latent_tokens}{ABS_END}\n"
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_prefix}],
        })

    return conversation


def run_inference(
    model,
    processor,
    conversation: List[Dict],
    max_new_tokens: int = 512,
    debug: bool = False,
) -> str:
    """
    Run inference on a single sample.

    Args:
        model: Model instance
        processor: Processor instance
        conversation: Conversation dict
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    # Prepare inputs
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=None,  # Images are handled in conversation
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to the same device as the model
    # When device_map="auto", model.device gives the first device
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    if debug:
        print(f"Input length: {len(inputs['input_ids'][0])}")
        print(f"Output length: {len(output_ids[0])}")
        print(f"Generated tokens: {len(output_ids[0]) - len(inputs['input_ids'][0])}")

    # Decode - only return newly generated tokens
    if output_ids is not None and "input_ids" in inputs and inputs["input_ids"] is not None:
        input_len = len(inputs["input_ids"][0])
        generated_ids = output_ids[0][input_len:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)

        if debug:
            print(f"Generated text length: {len(generated_text)}")
            print(f"Generated text: {generated_text[:200]}...")
    else:
        # Fallback: decode entire output
        generated_text = processor.decode(output_ids[0], skip_special_tokens=True) if output_ids is not None else ""

    return generated_text


def evaluate_gsm8k(
    model_path: str,
    test_jsonl: str,
    use_latent_mode: bool = False,
    latent_size: Optional[int] = None,
    force_prefix: bool = False,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate model on GSM8K test set.

    Args:
        model_path: Path to model checkpoint
        test_jsonl: Path to test JSONL file
        use_latent_mode: Whether model is trained with latent mode (for SFT models)
        latent_size: Number of latent tokens (only used if force_prefix=True)
        force_prefix: If True, pre-fill latent tokens in prompt (not recommended).
                      If False (default), let model generate latent tokens autoregressively.
        output_file: Optional path to save detailed results
        max_samples: Maximum number of samples to evaluate (for debugging)
        device: Device to run on

    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading model from {model_path}...")
    model, processor = load_model_and_processor(
        model_path,
        use_latent_mode=use_latent_mode,
        device=device,
    )

    print(f"Loading test data from {test_jsonl}...")
    with open(test_jsonl, 'r') as f:
        samples = [json.loads(line) for line in f]

    if max_samples:
        samples = samples[:max_samples]
        print(f"Evaluating first {max_samples} samples...")
    else:
        print(f"Evaluating {len(samples)} samples...")

    # Evaluate
    correct = 0
    results = []
    sample_idx = 0

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample["question"]
        gt_answer = sample["answer"]
        question_image_paths = sample.get("question_image_paths", [])

        # Build prompt
        conversation = build_inference_prompt(
            question=question,
            question_image_paths=question_image_paths,
            use_latent_mode=use_latent_mode,
            latent_size=latent_size,
            force_prefix=force_prefix,
        )

        # Run inference (debug first 3 samples)
        try:
            generated_text = run_inference(
                model=model,
                processor=processor,
                conversation=conversation,
                debug=(sample_idx < 3),  # Debug first 3 samples
            )

            sample_idx += 1

            # Extract answer
            pred_answer = extract_answer(generated_text)
            gt_answer_num = extract_answer(gt_answer)  # Extract number from ground truth

            # Compare
            is_correct = False
            if pred_answer and gt_answer_num:
                is_correct = normalize_answer(pred_answer) == normalize_answer(gt_answer_num)
                if is_correct:
                    correct += 1

            # Save result
            results.append({
                "question": question,
                "ground_truth": gt_answer,
                "ground_truth_answer": gt_answer_num,  # Save extracted answer for debugging
                "generated_text": generated_text,
                "predicted_answer": pred_answer,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"Error processing sample: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            results.append({
                "question": question,
                "ground_truth": gt_answer,
                "error": str(e),
                "correct": False,
            })

    # Calculate accuracy
    accuracy = correct / len(samples) * 100

    eval_results = {
        "model_path": model_path,
        "test_jsonl": test_jsonl,
        "use_latent_mode": use_latent_mode,
        "force_prefix": force_prefix,
        "latent_size": latent_size,
        "total_samples": len(samples),
        "correct": correct,
        "accuracy": accuracy,
    }

    print(f"\nResults:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Save detailed results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                "eval_results": eval_results,
                "detailed_results": results,
            }, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to {output_file}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GSM8K test set")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    # /netdisk/xiongyuan/data/gsm8k/test.jsonl
    parser.add_argument(
        "--test_jsonl",
        type=str,
        required=True,
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--use_latent_mode",
        action="store_true",
        help="Model is trained with latent mode (for SFT models). "
             "By default, model will generate latent tokens autoregressively.",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=None,
        help="Number of latent tokens (only used with --force_prefix). "
             "If not specified, model generates latent tokens autoregressively.",
    )
    parser.add_argument(
        "--force_prefix",
        action="store_true",
        help="Pre-fill latent tokens in prompt (NOT RECOMMENDED). "
             "By default (False), let model generate latent tokens autoregressively.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed results JSON",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on. Options: 'auto' (recommended), 'cuda', 'cuda:0', 'cuda:1', 'cpu'. Default: auto",
    )

    args = parser.parse_args()

    evaluate_gsm8k(
        model_path=args.model_path,
        test_jsonl=args.test_jsonl,
        use_latent_mode=args.use_latent_mode,
        latent_size=args.latent_size,
        force_prefix=args.force_prefix,
        output_file=args.output_file,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()


"""
uv run python evaluate_gsm8k.py     --model_path /netcache/huggingface/Qwen2.5-VL-7B-Instruct     --test_jsonl /netdisk/xiongyuan/data/gsm8k/test.jsonl     --output_file evaluation_results/original_model.json

uv run python evaluate_gsm8k.py     --model_path /netdisk/xiongyuan/VisualReasoning/SFT/out_stage3     --test_jsonl /netdisk/xiongyuan/data/gsm8k/test.jsonl     --use_latent_mode     --output_file evaluation_results/sft_model.json
"""
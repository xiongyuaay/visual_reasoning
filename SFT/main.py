from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import AutoProcessor
from trl import SFTConfig, SFTTrainer

from .data import load_gsm8k
from .collate import TokenBundle, add_special_tokens, collate_stage1, collate_stage2_or_3
from .patch_qwen25vl_latent import apply_qwen25vl_latent_patch
from .trainer import CustomTrainerLatentAlign


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["stage1", "stage2", "stage3"], default="stage1")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--latent_size", type=int, default=128, help="Maximum latent token count")
    p.add_argument("--min_latent_size", type=int, default=16, help="Minimum latent token count")
    p.add_argument("--latent_scale_factor", type=float, default=0.3, help="Compression ratio for dynamic latent sizing (0.3 = 30%% of original thought tokens)")
    p.add_argument("--teacher_latent_dir", type=str, default=None)
    p.add_argument("--align_weight", type=float, default=1.0)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fsdp", type=str, default="", help='Optional FSDP config string, e.g. "full_shard auto_wrap"')
    p.add_argument("--deepspeed", type=str, default="", help='Optional DeepSpeed config JSON path')
    return p.parse_args()


def build_hf_dataset(args: argparse.Namespace) -> Dataset:
    samples = load_gsm8k(
        jsonl_path=args.train_jsonl,
        limit=args.max_samples,
    )
    rows: List[Dict[str, Any]] = []
    for s in samples:
        rows.append(
            {
                "idx": s.idx,
                "question": s.question,
                "answer": s.answer,
                "question_image_paths": s.question_image_paths,
                "thought_image_paths": s.thought_image_paths,
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(int(args.seed))

    # Apply patch before model import/load so it picks up the patched modeling definitions.
    apply_qwen25vl_latent_patch()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    toks: TokenBundle = add_special_tokens(processor)

    # Prefer AutoModelForImageTextToText for MLLMs (e.g., Qwen2.5-VL). Fall back to CausalLM if needed.
    try:
        from transformers import AutoModelForImageTextToText as _AutoModel  # type: ignore
    except Exception:
        from transformers import AutoModelForCausalLM as _AutoModel  # type: ignore

    model = _AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    # Configure latent ids on model config for downstream logic in patched forward
    model.config.latent_token_id = int(toks.abs_pad_id)
    model.config.latent_start_id = int(toks.abs_start_id)
    model.config.latent_end_id = int(toks.abs_end_id)
    model.config.answer_start_pattern = toks.answer_start_pattern.tolist()

    ds = build_hf_dataset(args)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_steps=10,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=args.fsdp if args.fsdp else None,
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"],
            "limit_all_gathers": True,
        } if args.fsdp else None,
        deepspeed=args.deepspeed if args.deepspeed else None,
    )

    if args.stage == "stage1":
        trainer_cls = SFTTrainer
        collator = partial(
            collate_stage1,
            processor=processor,
            toks=toks,
            latent_size=args.latent_size,
            min_latent_size=args.min_latent_size,
            variable_latent=True,  # Default: random latent size per sample
            max_seq_len=args.max_seq_len,
        )
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=ds,
            data_collator=collator,
            processing_class=processor,
        )
    else:
        if not args.teacher_latent_dir:
            raise ValueError("--teacher_latent_dir is required for stage2/stage3")

        collator = partial(
            collate_stage2_or_3,
            processor=processor,
            toks=toks,
            latent_size=args.latent_size,
            min_latent_size=args.min_latent_size,
            variable_latent=(args.stage == "stage3"),
            max_seq_len=args.max_seq_len,
            include_thought_images=(args.stage == "stage2"),
            dynamic_latent_size=True,  # Default: dynamic latent based on thought image tokens
            latent_scale_factor=args.latent_scale_factor,
            teacher_latent_dir=args.teacher_latent_dir,
        )
        trainer = CustomTrainerLatentAlign(
            model=model,
            args=training_args,
            train_dataset=ds,
            data_collator=collator,
            processing_class=processor,
            teacher_latent_dir=args.teacher_latent_dir,
            align_weight=args.align_weight,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()


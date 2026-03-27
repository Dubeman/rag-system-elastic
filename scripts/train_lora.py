#!/usr/bin/env python3
"""
LoRA fine-tuning template for vision-language models (Phi-3.5-vision / SmolVLM).

Prepare a JSONL dataset with fields your chosen trainer expects (e.g. images,
question, answer), then implement the training loop below or use Hugging Face
TRL/SFT trainers.

Usage (after deps installed):
  python scripts/train_lora.py --output-dir adapters/lora-v2 --epochs 1
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="PEFT LoRA fine-tune (template)")
    parser.add_argument(
        "--model",
        default=os.getenv("VLM_MODEL_NAME", "microsoft/Phi-3.5-vision-instruct"),
        help="Base model id (HF hub or local path)",
    )
    parser.add_argument(
        "--output-dir",
        default="adapters/lora-v2",
        help="Where to save LoRA adapter weights",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        from peft import LoraConfig, TaskType  # type: ignore
    except ImportError:
        print(
            "Missing dependencies. Install: peft torch transformers accelerate",
            file=sys.stderr,
        )
        return 1

    _ = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    readme = os.path.join(args.output_dir, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "LoRA adapter placeholder.\n"
            f"Intended base model: {args.model}\n"
            "Implement dataset loading and trainer; keep adapters versioned for regression tests.\n"
        )

    print(
        f"Template complete. Configure dataset + trainer, then save adapter to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

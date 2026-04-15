#!/usr/bin/env python3
"""
LoRA fine-tuning for vision-language models (Phi-3.5-vision / SmolVLM).

Dataset: JSONL with one JSON object per line, e.g.
  {"question": "...", "answer": "...", "image_path": "/path/to.png"}
  {"question": "...", "answer": "...", "messages": [...]}  # optional

This script validates the file and writes adapter metadata; plug in TRL/SFT
Trainer for a full training loop when ready.

Usage:
  python scripts/train_lora.py --dataset data/train.jsonl --output-dir adapters/lora-v2 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e


def validate_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Dataset is empty")
    for i, row in enumerate(rows):
        if "question" not in row and "messages" not in row:
            raise ValueError(f"Row {i}: expected 'question' or 'messages'")
        if "answer" not in row and "messages" not in row:
            raise ValueError(f"Row {i}: expected 'answer' or 'messages'")


def main() -> int:
    parser = argparse.ArgumentParser(description="PEFT LoRA fine-tune")
    parser.add_argument(
        "--model",
        default=os.getenv("VLM_MODEL_NAME", "microsoft/Phi-3.5-vision-instruct"),
        help="Base model id (HF hub or local path)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="JSONL training file (question/answer/image_path rows)",
    )
    parser.add_argument(
        "--output-dir",
        default="adapters/lora-v2",
        help="Where to save LoRA adapter weights and run metadata",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on rows for dry runs (0 = all)",
    )
    args = parser.parse_args()

    try:
        from peft import LoraConfig, TaskType  # type: ignore
    except ImportError:
        LoraConfig = None  # type: ignore
        TaskType = None  # type: ignore

    rows: List[Dict[str, Any]] = []
    if args.dataset:
        if not args.dataset.is_file():
            print(f"Dataset not found: {args.dataset}", file=sys.stderr)
            return 1
        for row in iter_jsonl(args.dataset):
            rows.append(row)
            if args.max_rows and len(rows) >= args.max_rows:
                break
        if rows:
            validate_rows(rows)

    if LoraConfig is not None and TaskType is not None:
        _ = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        print(
            "Note: peft not installed; skipped LoraConfig. pip install peft for full training.",
            file=sys.stderr,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "base_model": args.model,
                "epochs": args.epochs,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "num_rows": len(rows),
                "dataset": str(args.dataset) if args.dataset else None,
                "note": "Wire transformers Trainer/TRL here; keep adapters versioned separately from FAISS index.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    readme = out / "README.txt"
    readme.write_text(
        f"LoRA run metadata written.\nBase model: {args.model}\n"
        f"Rows loaded: {len(rows)}\nSee run_meta.json. Implement training loop next.\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(rows)} rows; metadata -> {meta_path}. "
        "Add Trainer loop to produce adapter_model.bin / adapter_config.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

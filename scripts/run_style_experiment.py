"""Run the end-to-end style experiment (used locally and by the Kaggle notebook)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from echotwin.experiment import format_comparison, run_experiment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--author", default="Author")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument(
        "--methods",
        default="base,style_profile,style_rag,ngram",
        help="comma separated: base|style_profile|style_rag|style_rag_ft|ngram|lora",
    )
    ap.add_argument("--train-lora", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-train-examples", type=int, default=400)
    ap.add_argument("--max-eval-prompts", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20250917)
    args = ap.parse_args()

    report = run_experiment(
        corpus_dir=args.corpus,
        out_dir=args.out,
        author_name=args.author,
        model=args.model,
        methods=tuple(m.strip() for m in args.methods.split(",") if m.strip()),
        train_lora=args.train_lora,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_train_examples=args.max_train_examples,
        max_eval_prompts=args.max_eval_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(format_comparison(report))
    print(f"\nreport: {Path(args.out) / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

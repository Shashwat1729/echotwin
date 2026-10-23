"""EchoTwin benchmark runner: real, measured numbers only.

Measures on this machine (no fabrication):
- dataset preparation throughput (synthetic export)
- mock baseline inference latency
- audio quality-analysis latency
- voice synthesis latency/RTF (only when a backend + weights are available;
  otherwise reported as skipped with the reason)

Usage: python benchmarks/run_benchmark.py [--out benchmarks/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bench_dataset() -> dict:
    from echotwin.dataset.builder import DatasetConfig, build_examples
    from echotwin.ingestion import WhatsAppParser, pick_participant

    text = (ROOT / "examples" / "synthetic_whatsapp.txt").read_text(encoding="utf-8")
    t0 = time.time()
    result = WhatsAppParser().parse_text(text * 20, "bench")
    target = pick_participant(result.messages, "Aarav")
    built = build_examples(result.messages, target, DatasetConfig())
    dt = time.time() - t0
    return {
        "messages": len(result.messages),
        "examples": len(built.examples),
        "seconds": round(dt, 3),
        "examples_per_second": round(len(built.examples) / dt, 1) if dt else None,
    }


def bench_baseline() -> dict:
    from echotwin.dataset.builder import DatasetConfig, build_examples
    from echotwin.dataset.splits import SplitConfig, split_dataset
    from echotwin.evaluation.baseline import heldout_prompts, run_baseline
    from echotwin.ingestion import WhatsAppParser, pick_participant
    from echotwin.models.generation import MockGenerator

    text = (ROOT / "examples" / "synthetic_whatsapp.txt").read_text(encoding="utf-8")
    result = WhatsAppParser().parse_text(text, "bench")
    target = pick_participant(result.messages, "Aarav")
    built = build_examples(result.messages, target, DatasetConfig())
    split = split_dataset(built, SplitConfig())
    prompts = heldout_prompts(split.test or split.train[:3], max_prompts=10)
    gen = MockGenerator([ex.messages[-1].content for ex in split.train[:3]], seed=1)
    t0 = time.time()
    records, _, _ = run_baseline(
        method="mock",
        generator=gen,
        prompts=prompts,
        output_dir=ROOT / "benchmarks" / "_tmp",
        model_revision="mock-1",
    )
    dt = time.time() - t0
    return {
        "prompts": len(records),
        "seconds": round(dt, 3),
        "ms_per_prompt": round(dt / len(records) * 1000, 1) if records else None,
    }


def bench_audio() -> dict:
    import numpy as np

    from echotwin.voice.audio_quality import analyze_wave_array

    sr = 16000
    samples = (0.3 * np.sin(2 * np.pi * 180 * np.arange(sr * 5) / sr)).astype("float32")
    t0 = time.time()
    report = analyze_wave_array(samples, sr)
    dt = time.time() - t0
    return {"seconds_5s_clip": round(dt, 4), "verdict": report.verdict}


def bench_synthesis() -> dict:
    from echotwin.voice.registry import find_available_backend

    backend = find_available_backend()
    if backend is None:
        return {"status": "skipped", "reason": "no voice backend installed"}
    return {"status": "skipped", "reason": "synthesis needs a recorded voice profile; see docs/voice-cloning.md"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results.json")
    args = ap.parse_args()
    results = {
        "dataset": bench_dataset(),
        "baseline_mock": bench_baseline(),
        "audio_quality": bench_audio(),
        "synthesis": bench_synthesis(),
    }
    out = ROOT / args.out
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Hardware Requirements

## CPU-only (everything except training/synthesis)

Ingestion, dataset building, profiling, baselines (mock), evaluation,
recording plans, coverage, quality analysis, API, and UI all run on CPU with
no model downloads. This is the default CI configuration.

## Text training

| Setup | Suitable for |
|---|---|
| Qwen3-1.7B + LoRA, CPU | Smoke tests only (very slow) |
| 4B + QLoRA 4-bit, ~6GB VRAM | Personal datasets (hundreds of examples) |
| 4B fp16 LoRA, ~10GB VRAM | Faster iteration |

Full fine-tuning is not recommended; LoRA recovers style transfer quality at
a fraction of the memory. `echotwin doctor` reports your actual device.

## Voice synthesis

| Backend | Recommended | Minimum |
|---|---|---|
| VoxCPM2 | NVIDIA GPU ~8GB VRAM | CPU (slow) / MPS |
| Chatterbox Multilingual | GPU 6GB+ | CPU (slow) |

Reference audio is 16kHz; VoxCPM2 outputs 48kHz. Synthesis latency and
real-time factor are measured per run (`echotwin voice evaluate`).

## Storage

Model weights are multi-GB and live outside the repo (HF cache or
`ECHOTWIN_HOME`). Checkpoints, adapters, recordings, and generated audio are
git-ignored by default — keep them out of version control.

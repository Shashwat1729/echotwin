"""Training package: SFT/LoRA/QLoRA via transformers+PEFT, optional TRL paths.

Every heavy import happens inside the functions so CPU-only installs can use the
rest of EchoTwin unimpeded. QLoRA validation is strict: unsupported hardware
raises rather than silently switching method.
"""

from __future__ import annotations

from .sft import LoRAConfig, TrainingConfig, format_for_sft, resume_sft, run_sft

__all__ = ["LoRAConfig", "TrainingConfig", "format_for_sft", "resume_sft", "run_sft"]

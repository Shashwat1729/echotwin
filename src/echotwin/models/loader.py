"""Model loader with strict lazy imports.

Heavy dependencies (torch, transformers, peft, bitsandbytes) are imported inside
``load_model`` / adapter functions so that ``echotwin --help``, ingestion, dataset
building, profiling, voice planning and all core tests work on a CPU-only install
without any model download.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..exceptions import ModelNotAvailableError, UnsupportedCapabilityError
from ..models.registry import ModelSpec


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    spec: ModelSpec
    device: str
    quantized: bool = False
    dtype: str = "float32"
    chat_template: str | None = None
    notes: list[str] = field(default_factory=list)


class TextGenerator(Protocol):
    def generate(
        self, messages: list[dict], *, max_new_tokens: int = 128, seed: int | None = None
    ) -> str: ...


def _require(module: str, extra: str) -> Any:
    try:
        return __import__(module)
    except (ImportError, OSError) as e:  # pragma: no cover - depends on install
        reason = "not installed" if isinstance(e, ImportError) else "installed but unloadable"
        raise ModelNotAvailableError(
            f"'{module}' is {reason} ({type(e).__name__}). Install extra: {extra}"
        ) from e


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", False) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def hardware_report() -> dict:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": detect_device(),
        "cuda_available": False,
        "cuda_device": None,
        "cpu_count": None,
        "total_ram_gb": None,
        "bitsandbytes_available": False,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import psutil  # optional

        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception:
        pass
    try:
        __import__("bitsandbytes")
        info["bitsandbytes_available"] = True
    except Exception:
        pass
    return info


def load_model(
    spec: ModelSpec,
    *,
    quantize: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load tokenizer + causal LM. ``quantize`` in {None, '4bit'}.

    4-bit (QLoRA) requires CUDA + bitsandbytes + accelerate; otherwise we raise
    UnsupportedCapabilityError instead of silently falling back.
    """
    transformers = _require("transformers", "train")
    torch = _require("torch", "train")

    device = detect_device()
    quantized = False
    quant_kwargs: dict[str, Any] = {}
    if quantize == "4bit":
        if device != "cuda":
            raise UnsupportedCapabilityError(
                f"4-bit quantization (QLoRA) requires a CUDA GPU; detected device='{device}'. "
                "Run on GPU or use fp16/bf16 LoRA instead."
            )
        try:
            __import__("bitsandbytes")
        except ImportError as e:
            raise UnsupportedCapabilityError(
                "4-bit quantization requires 'bitsandbytes' (pip install bitsandbytes) "
                "and a CUDA GPU."
            ) from e
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        quantized = True

    notes: list[str] = []
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, **quant_kwargs}
    if dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(spec.hf_model_id)
    except Exception as e:
        raise ModelNotAvailableError(
            f"Could not download/load tokenizer for '{spec.hf_model_id}': {e}"
        ) from e

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        notes.append("pad_token set to eos_token")

    has_template = getattr(tokenizer, "chat_template", None) is not None
    if spec.chat_template_expected and not has_template:
        notes.append("no chat template on tokenizer; will fall back to plain prompt formatting")

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(spec.hf_model_id, **kwargs)
    except Exception as e:
        raise ModelNotAvailableError(
            f"Could not download/load model '{spec.hf_model_id}': {e}"
        ) from e

    if device in {"cuda", "mps"}:
        pass  # device_map/manual .to handled by caller for adapters
    resolved_dtype = str(next(model.parameters()).dtype)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        device=device,
        quantized=quantized,
        dtype=resolved_dtype,
        chat_template=getattr(tokenizer, "chat_template", None),
        notes=notes,
    )


def apply_lora(model, tokenizer, lora_cfg) -> tuple[Any, list[str]]:
    """Apply PEFT LoRA with explicit module-target validation."""
    _require("peft", "train")
    from peft import LoraConfig, get_peft_model

    target_modules = resolve_target_modules(model, lora_cfg.target_modules)
    if not target_modules:
        raise UnsupportedCapabilityError(
            "LoRA target-module patterns matched no modules in this architecture. "
            f"Model={type(model).__name__}; patterns={lora_cfg.target_modules}"
        )
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=target_modules,
        bias=lora_cfg.bias,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    return peft_model, target_modules


def resolve_target_modules(model, patterns: list[str]) -> list[str]:
    """Match LoraConfig-style module name suffixes against the model's modules."""
    existing = {name for name, _ in model.named_modules()}
    matched: list[str] = []
    for pattern in patterns:
        rx = _compile_module_pattern(pattern)
        hits = sorted(n for n in existing if rx.search(n))
        matched.extend(hits)
    return sorted(set(matched))


def _compile_module_pattern(pattern: str):
    import re

    escaped = re.escape(pattern).replace(re.escape(".*"), ".*")
    return re.compile(escaped)


def load_model_with_adapter(spec: ModelSpec, adapter_dir: str | Path, *, quantize: str | None = None):
    """Load a base model and attach a trained LoRA adapter for inference.

    The adapter is a PEFT directory produced by ``run_sft``. We load the base
    weights first (shared helper) and then attach the adapter, so inference
    uses exactly the same code path as training.
    """
    _require("peft", "train")  # fail early with the documented extra name
    from peft import PeftModel

    loaded = load_model(spec, quantize=quantize)
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise ModelNotAvailableError(f"Adapter directory not found: {adapter_path}")
    try:
        loaded.model = PeftModel.from_pretrained(loaded.model, str(adapter_path))
    except Exception as e:
        raise ModelNotAvailableError(
            f"Could not attach LoRA adapter from {adapter_path}: {e}. "
            "Check that the adapter was trained on the same base model."
        ) from e
    loaded.notes.append(f"attached LoRA adapter: {adapter_path.name}")
    return loaded


def unload(model) -> None:
    try:
        import gc

        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass

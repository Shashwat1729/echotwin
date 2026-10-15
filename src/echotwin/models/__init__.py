"""Models package."""

from .generation import (
    GeneratorConfig,
    MockGenerator,
    TextGenerator,
    TransformersGenerator,
    load_predictions,
    save_predictions,
)
from .loader import (
    LoadedModel,
    apply_lora,
    detect_device,
    hardware_report,
    load_model,
    resolve_target_modules,
    unload,
)
from .registry import DEFAULT_MODEL, REGISTRY, ModelSpec, get_model_spec, list_models

__all__ = [
    "DEFAULT_MODEL",
    "REGISTRY",
    "GeneratorConfig",
    "LoadedModel",
    "MockGenerator",
    "ModelSpec",
    "TextGenerator",
    "TransformersGenerator",
    "apply_lora",
    "detect_device",
    "get_model_spec",
    "hardware_report",
    "list_models",
    "load_model",
    "load_predictions",
    "resolve_target_modules",
    "save_predictions",
    "unload",
]

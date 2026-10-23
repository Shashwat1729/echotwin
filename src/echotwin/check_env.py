"""Environment diagnostics used by 'echotwin doctor'."""

from __future__ import annotations

import importlib
import platform
import shutil
import sys


def _module_status(module: str) -> str:
    try:
        m = importlib.import_module(module)
        return str(getattr(m, "__version__", "installed"))
    except ModuleNotFoundError:
        return "missing"
    except ImportError:
        return "missing"
    except Exception as e:
        return f"blocked ({type(e).__name__})"


def doctor_report() -> dict:
    lines: list[str] = []
    healthy = True

    lines.append(f"python: {sys.version.split()[0]} ({platform.platform()})")
    if sys.version_info < (3, 11):  # noqa: UP036 - doctor must verify the live runtime, not packaging metadata
        lines.append("[red]Python <3.11 is not supported[/red]")
        healthy = False

    core = ["pydantic", "yaml", "typer", "fastapi", "numpy", "regex"]
    # A core dependency that is *present but broken* (imports and raises - e.g.
    # a blocked native DLL) is just as fatal as a missing one, so anything that
    # is not a working import counts against health.
    broken_core: list[str] = []
    for m in core:
        status = _module_status(m)
        if status == "missing":
            broken_core.append(f"{m} (missing)")
        elif status.startswith("blocked"):
            broken_core.append(f"{m} ({status})")
    lines.append("core deps: " + ("OK" if not broken_core else f"BROKEN {broken_core}"))
    if broken_core:
        healthy = False

    optional = {
        "torch": "train",
        "transformers": "train",
        "peft": "train",
        "trl": "train",
        "soundfile": "voice",
        "scipy": "voice",
        "gradio": "ui",
        "langdetect": "lang",
        "resemblyzer": "voice-eval",
        "faster_whisper": "voice-eval",
    }
    for mod, extra in optional.items():
        status = _module_status(mod)
        if status == "missing":
            flag = "missing"
        elif status.startswith("blocked"):
            flag = status
        else:
            flag = "OK"
        lines.append(f"optional {mod:12s}: {flag} ({extra})")

    # backends
    try:
        from .voice.registry import all_backend_statuses

        for b in all_backend_statuses():
            caps = b.get("capabilities", {})
            lines.append(f"backend {b['name']:22s}: available={caps.get('available', False)}")
    except Exception as e:
        lines.append(f"backend registry error: {e}")
        healthy = False

    # storage
    try:
        from .settings import get_settings
        from .storage.paths import home_summary

        s = get_settings()
        s.ensure_dirs()
        info = home_summary()
        lines.append(f"home: {info['home']} (writable={info['writable']})")
        lines.append(f"db: {info['db']}")
        if not info["writable"]:
            lines.append("[red]ECHOTWIN_HOME is not writable[/red]")
            healthy = False
    except Exception as e:
        lines.append(f"[red]storage error: {e}[/red]")
        healthy = False

    # hardware for training
    try:
        import torch

        cuda = torch.cuda.is_available()
        lines.append(f"torch: {torch.__version__} cuda_available={cuda}")
        if cuda:
            lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
        else:
            lines.append("gpu: none (training runs on CPU; QLoRA unavailable)")
    except ModuleNotFoundError:
        lines.append("torch: missing (install extra: .[train])")
    except ImportError:
        lines.append("torch: missing (install extra: .[train])")
    except Exception as e:
        lines.append(f"torch: blocked ({type(e).__name__}; install extra: .[train])")

    if shutil.which("git") is None:
        lines.append("git: missing (optional)")
    else:
        lines.append("git: OK")

    # GPU-less hints
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "  - Ingestion/dataset/profiling/voice-planning run fine on CPU without model downloads."
    )
    lines.append(
        "  - Text training needs the '[train]' extra + a model download (GPU strongly recommended)."
    )
    lines.append("  - Voice synthesis needs a backend (pip install voxcpm) + weights (about 4GB).")
    return {"lines": lines, "healthy": healthy}

"""EchoTwin command line interface.

Heavy dependencies (torch/transformers/soundfile/voxcpm) are imported lazily
inside the commands that need them, so ``echotwin --help`` and all data-plane
commands work on a minimal install.
"""

from __future__ import annotations

from . import admin, data, evaluate, train, voice  # noqa: F401  (register commands)
from ._common import app
from .admin import main

__all__ = ["app", "main"]

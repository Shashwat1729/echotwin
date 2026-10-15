"""Explicit model downloader: `python scripts/download_models.py --model qwen3-4b-it-2507`.

Nothing in EchoTwin downloads weights implicitly; run this when you are ready
for training or synthesis. Requires the [train] extra (huggingface_hub).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download an EchoTwin model explicitly.")
    ap.add_argument("--model", default="qwen3-4b-it-2507", help="registry name or org/model id")
    ap.add_argument("--voice", action="store_true", help="download the voice backend model instead")
    args = ap.parse_args()

    if args.voice:
        from echotwin.voice.registry import get_backend

        backend = get_backend("voxcpm")
        model_id = getattr(backend, "model_id", "openbmb/VoxCPM2")
    else:
        from echotwin.models.registry import get_model_spec

        model_id = get_model_spec(args.model).hf_model_id

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Install the [train] extra first.", file=sys.stderr)
        return 2

    print(f"Downloading {model_id} ...")
    path = snapshot_download(repo_id=model_id)
    print(f"Cached at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

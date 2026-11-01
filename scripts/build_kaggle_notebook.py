"""Generate the Kaggle notebook that runs the real LoRA style-transfer experiment.

The notebook is written by this script so it stays in sync with the repository
and is reproducible (no hand-edited JSON). It is pushed with
``kaggle kernels push`` and reads EchoTwin from a PRIVATE Kaggle dataset, so
no credential is ever embedded in the notebook file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

KERNEL_ID = "shashwat1729/echotwin-lora-style-transfer"
# Kaggle slugifies the title into the kernel id, so the title must resolve to
# KERNEL_ID's slug or the push is rejected.
KERNEL_TITLE = "Echotwin Lora Style Transfer"
SOURCE_DATASET = "shashwat1729/echotwin-source"
PERSONAL_DATASET = "shashwat1729/echotwin-personal"

# TPU is deliberately NOT a tier: the whole training/eval stack is PyTorch;
# a TPU kernel needs torch_xla end to end (optimizer + generation loop), which
# is a porting project -- not a metadata flag.
TIERS: dict[str, dict[str, Any]] = {
    "gpu": {
        "accelerator": "GPU",
        "gpu_type": "T4",
        "enable_gpu": True,
        "model": "qwen3-4b-it-2507",
        "epochs": 2,
        "lora_rank": 16,
        "learning_rate": "1e-4",
        "max_train_examples": 700,
        "max_eval_prompts": 60,
        "max_new_tokens": 56,
        # GPU tier skips the CPU-torch preinstall entirely (CUDA build preinstalled).
        "cpu_torch": None,
        "cpu_torchvision": None,
        "session_note": "T4 GPU (the authoritative run; ~30 GPU-hours/week quota)",
    },
    "personal": {
        # Your own chats + voice from the PRIVATE echotwin-personal dataset.
        "accelerator": "GPU",
        "gpu_type": "T4",
        "enable_gpu": True,
        "model": "qwen3-1.7b",
        "epochs": 2,
        "lora_rank": 16,
        "learning_rate": "1e-4",
        "max_train_examples": 2000,
        "max_eval_prompts": 100,
        "max_new_tokens": 56,
        "cpu_torch": None,
        "cpu_torchvision": None,
        "personal": True,
        "session_note": "T4 GPU, your private chats; adapter + voice samples saved to /kaggle/working",
    },
    "cpu": {
        "accelerator": "CPU",
        "gpu_type": None,
        "enable_gpu": False,
        # The smallest registered spec: proven LoRA plumbing on free CPU hours.
        # Its numbers show the training path works, not headline quality.
        "model": "qwen3-0.6b",
        "epochs": 2,
        "lora_rank": 8,
        "learning_rate": "2e-4",
        "max_train_examples": 400,
        "max_eval_prompts": 40,
        "max_new_tokens": 48,
        # torch 2.9.0 pair that exists as Windows-CPU wheels (open constraints
        # resolve the newest release to an sdist-only index for CPU).
        "cpu_torch": "2.9.0",
        "cpu_torchvision": "0.24.0",
        "session_note": "4-core CPU - fits the 12h Kaggle wall clock comfortably",
    },
}


def _cells(tier: dict[str, Any]) -> list[tuple[str, str]]:
    model = tier["model"]
    personal = bool(tier.get("personal"))
    corpus = "personal_chats" if personal else "sample_corpus/daily_dialog/chats"
    author = f'--author "{os.environ.get("ECHOTWIN_AUTHOR", "Me")}"' if personal else ""
    return [
        (
            "markdown",
            f"""# EchoTwin — real-corpus LoRA style transfer (Kaggle {tier['accelerator']})

This notebook runs the **actual EchoTwin pipeline** on a **real, public,
multi-conversation corpus** and reports measured numbers:

1. install EchoTwin from a private Kaggle dataset (no secrets in this file),
2. fetch a real chat corpus and convert it to WhatsApp-style exports,
3. import it with the production parser, pick the author, build the dataset,
   split by conversation (leakage-resistant),
4. train a LoRA adapter (SFT with verified target masking),
5. evaluate **base / style-profile / style-rag / LoRA** on the *same* held-out
   prompts, and report style similarity, prompt responsiveness, contamination
   and failures.

**Honesty note:** every number below was produced by running the code in this
notebook. The corpus author is a *role* in a public corpus, not one individual,
so the result shows that the method captures a consistent speaker register -
it is not a claim about any specific person.
""",
    ),
    (
        "code",
        """import importlib.metadata as _md
# Pin the already-imported numpy/pandas: upgrading them under a live kernel breaks the ABI.
open("/tmp/pins.txt", "w").write("\\n".join(f"{n}=={_md.version(n)}" for n in ("numpy", "pandas")))
!pip install -q -c /tmp/pins.txt --upgrade "transformers>=5.0" "peft>=0.21" "trl>=0.15" \\
    accelerate datasets huggingface_hub pyarrow soundfile scipy pandas pytest
!pip uninstall -y -q torchao  # preinstalled 0.10 is rejected by peft; unused here
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("transformers", transformers.__version__, "peft", peft.__version__)
""",
    ),
    *(
        [
            (
                "code",
                f"""# 2) CPU tier: Kaggle CPU images ship no torch build, and pyproject's open
#    constraint resolves Windows to an index without CPU wheels - pin the
#    2.9.0 family from the official CPU wheel index first.
!pip install -q torch=={tier["cpu_torch"]} torchvision=={tier["cpu_torchvision"]} torchaudio=={tier["cpu_torch"]} --index-url https://download.pytorch.org/whl/cpu
""",
            )
        ]
        if tier.get("cpu_torch")
        else []
    ),
    (
        "code",
        f"""# Install EchoTwin from the PRIVATE dataset mirror (no credential in this file)
!mkdir -p /kaggle/working/src
!kaggle datasets download -d {SOURCE_DATASET} -p /kaggle/working/src --unzip --force
!cd /kaggle/working/src && for z in *.zip; do [ -f "$z" ] && unzip -o "$z"; done; ls /kaggle/working/src
!pip install -q --ignore-requires-python -c /tmp/pins.txt -e "/kaggle/working/src[voice,train,lang,dev]"
import sys; sys.path.insert(0, "/kaggle/working/src/src")  # .pth files are read at startup only
import echotwin
print("echotwin", echotwin.__version__, "from", echotwin.__file__)
""",
    ),
    (
        "code",
        ""
        + (
            """import glob, os, shutil
os.makedirs("/kaggle/working/personal_chats", exist_ok=True)
files = glob.glob("/kaggle/input/echotwin-personal/chats/*_chat.txt")
for f in files:
    shutil.copy(f, "/kaggle/working/personal_chats/")
print("chats:", len(files))
print(sorted(os.listdir("/kaggle/input/echotwin-personal/voice"))[:3])
"""
            if personal
            else """# 1) Fetch a REAL multi-conversation chat corpus (DailyDialog mirror) and
# convert it to WhatsApp-style exports that the production importer reads.
!cd /kaggle/working && python src/scripts/fetch_public_corpus.py --max-conversations 600
!ls /kaggle/working/sample_corpus/daily_dialog/chats | head -3
"""
        ),
    ),
    (
        "code",
        f"""# 2) Run the experiment: dataset -> conversation split -> LoRA -> evaluation.
#    Every method sees the SAME held-out prompts. ({tier['session_note']})
!cd /kaggle/working && python src/scripts/run_style_experiment.py \\
    --corpus {corpus} {author} \\
    --out artifacts/experiments/kaggle \\
    --model {model} \\
    --methods base,style_profile,style_rag,ngram,lora,style_rag_ft \\
    --train-lora --epochs {tier['epochs']} --lora-rank {tier['lora_rank']} \\
    --learning-rate {tier['learning_rate']} \\
    --max-train-examples {tier['max_train_examples']} \\
    --max-eval-prompts {tier['max_eval_prompts']} --max-new-tokens {tier['max_new_tokens']}
""",
    ),
    (
        "code",
        """# 3) Show the report and the generated replies
import json, pathlib
report = json.loads(pathlib.Path("/kaggle/working/artifacts/experiments/kaggle/report.json").read_text())
print("corpus:", report["corpus"])
print("splits:", report["splits"])
for m in report["methods"]:
    s = m.get("style") or {}
    print(f"{m['method']:<14} cos3={s.get('char_ngram_cosine_3')} "
          f"len_ratio={s.get('length_mean_ratio')} "
          f"distinct={(m.get('prompt_discrimination') or {}).get('distinct_ratio')} "
          f"uptake={(m.get('prompt_uptake') or {}).get('mean_uptake')} "
          f"contam={(m.get('contamination') or {}).get('verdict')} failed={m['n_failed']}")
print()
for name in ("plain", "style_rag", "lora"):
    p = pathlib.Path(f"/kaggle/working/artifacts/experiments/kaggle/predictions_{name}.jsonl")
    if not p.exists():
        continue
    print(f"--- {name} ---")
    for line in p.read_text(encoding="utf-8").splitlines()[:3]:
        r = json.loads(line)
        print("  prompt:", (r.get("context") or [""])[-1][:70] if r.get("context") else "")
        print("  model :", (r.get("generated") or "")[:100])
        print("  human :", (r.get("reference") or "")[:100])
""",
    ),
    *(
        [
            (
                "code",
                """# 5) VOICE CLONING (VoxCPM). Runs only if voice consent was recorded in the
#    dataset (granted by the speaker in the EchoTwin studio) - otherwise skipped.
import json, pathlib, shutil, subprocess, sys
import glob
_hits = glob.glob("/kaggle/input/**/voice/manifest.json", recursive=True)
vin = pathlib.Path(_hits[0]).parent.parent if _hits else pathlib.Path("/kaggle/input/echotwin-personal")
consent = vin / "voice_consent.json"
if not consent.exists():
    print("SKIPPED: no voice_consent.json in the dataset - voice consent has not been granted.")
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--ignore-requires-python", "voxcpm", "soundfile"], check=True)
    from voxcpm import VoxCPM
    import soundfile as sf
    manifest = json.loads((vin / "voice" / "manifest.json").read_text(encoding="utf-8"))
    # reference: the clip closest to 12 s that has a script-aligned transcript
    ref = min((m for m in manifest if m["transcript"]), key=lambda m: abs(m["seconds"] - 12))
    ref_path = str(vin / "voice" / ref["file"])
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    out = pathlib.Path("/kaggle/working/voice_out"); out.mkdir(exist_ok=True)
    tests = {
        "hinglish": "Yaar kal milte hain, main ek naya idea lekar aa raha hoon, tum bas time pe aa jaana.",
        "english": "This is a short test of my cloned voice, speaking in a calm and natural way.",
        "hindi": "नमस्ते, यह मेरी आवाज़ का एक छोटा सा परीक्षण है।",
    }
    for name, text in tests.items():
        for mode in ("reference", "continuation"):
            kw = {"text": text, "reference_wav_path": ref_path}
            if mode == "continuation":
                kw.update(prompt_wav_path=ref_path, prompt_text=ref["transcript"])
            wav = model.generate(**kw)
            sf.write(str(out / f"{name}_{mode}.wav"), wav, model.tts_model.sample_rate)
            print("wrote", f"{name}_{mode}.wav")
    shutil.copy(ref_path, out / "reference_used.wav")
""",
            )
        ]
        if personal
        else []
    ),
    (
        "code",
        """# 4) Persist the results as a Kaggle dataset so they can be pulled back
#    into the repository and cited in the docs.
!cd /kaggle/working && mkdir -p echotwin-results && cp -r artifacts/experiments/kaggle/* echotwin-results/ 2>/dev/null || true
!ls -la /kaggle/working/echotwin-results | head
""",
    ),
    (
        "markdown",
        """## Interpreting the numbers

* **cos3** - character 3-gram cosine between the model's replies and the
  author's real held-out replies. Higher = closer surface style.
* **len_ratio** - mean generated length / mean author length. ~1.0 means the
  model matches the author's message length.
* **distinct** - share of *different* replies across prompts. A low value means
  the model ignores the prompt and repeats a canned answer.
* **uptake** - share of the prompt's content words echoed in the reply
  (a responsiveness proxy, not a correctness measure).
* **contam** - memorization verdict: `clean` / `some_overlap` / `high_overlap`.
* **failed** - generations that errored. A method with failures is not
  comparable on the other columns.

Caveats: the corpus is public dialogue (not WhatsApp, not Hinglish), the
author is a role rather than one person, and a single seed per prompt means
small differences are noise. The report JSON records all of this.
""",
    ),
]


def build_notebook(tier: dict[str, Any]) -> dict:
    cells = []
    for kind, source in _cells(tier):
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source.splitlines(keepends=True),
                }
            )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": _kaggle_meta(tier),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _kaggle_meta(tier: dict[str, Any]) -> dict[str, Any]:
    # Kaggle accepts both spellings depending on context; gpu_type only exists
    # when a GPU is actually enabled, otherwise pushes fail validation.
    meta: dict[str, Any] = {
        "accelerator": tier["accelerator"],
        "enable_gpu": tier["enable_gpu"],
    }
    if tier.get("gpu_type"):
        meta["gpu_type"] = tier["gpu_type"]
    meta["isPrivate"] = True
    return meta


def build_kernel_metadata(tier: dict[str, Any]) -> dict:
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "echotwin_lora.ipynb",
        "kernel_type": "notebook",
        "language": "python",
        "is_private": True,
        **_kaggle_meta(tier),
        "dataset_sources": [SOURCE_DATASET, *([PERSONAL_DATASET] if tier.get("personal") else [])],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Kaggle LoRA kernel notebook.")
    ap.add_argument(
        "--accelerator",
        choices=("gpu", "cpu", "personal"),
        default="gpu",
        help="gpu = qwen3-4b-it-2507 on T4 (authoritative); personal = your private chats + voice on T4; cpu = qwen3-0.6b, "
        "smaller train/eval limits that fit a 12h CPU session. TPU is not "
        "offered: the pipeline is PyTorch and has no torch_xla port yet.",
    )
    args = ap.parse_args()
    tier = TIERS[args.accelerator]
    out_dir = REPO_ROOT / "notebooks" / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / "echotwin_lora.ipynb"
    nb_path.write_text(json.dumps(build_notebook(tier), indent=1), encoding="utf-8")
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(build_kernel_metadata(tier), indent=2), encoding="utf-8"
    )
    print(f"wrote {nb_path}")
    print(f"wrote {out_dir / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()

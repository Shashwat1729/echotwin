"""Generate the Kaggle notebook that runs the real LoRA style-transfer experiment.

The notebook is written by this script so it stays in sync with the repository
and is reproducible (no hand-edited JSON). It is pushed with
``kaggle kernels push`` and reads EchoTwin from a PRIVATE Kaggle dataset, so
no credential is ever embedded in the notebook file.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

KERNEL_ID = "shashwat1729/echotwin-lora-style-transfer"
# Kaggle slugifies the title into the kernel id, so the title must resolve to
# KERNEL_ID's slug or the push is rejected.
KERNEL_TITLE = "Echotwin Lora Style Transfer"
SOURCE_DATASET = "shashwat1729/echotwin-source"

CELLS = [
    (
        "markdown",
        """# EchoTwin — real-corpus LoRA style transfer (Kaggle GPU)

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
        """!pip install -q --upgrade "transformers>=5.0" "peft>=0.21" "trl>=0.15" \\
    accelerate datasets huggingface_hub pyarrow soundfile scipy numpy pandas pytest
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("transformers", transformers.__version__, "peft", peft.__version__)
""",
    ),
    (
        "code",
        f"""# Install EchoTwin from the PRIVATE dataset mirror (no credential in this file)
!mkdir -p /kaggle/working/src
!kaggle datasets download -d {SOURCE_DATASET} -p /kaggle/working/src --unzip --force
!pip install -q -e "/kaggle/working/src[voice,train,lang,dev]"
import echotwin
print("echotwin", echotwin.__version__, "from", echotwin.__file__)
""",
    ),
    (
        "code",
        """# 1) Fetch a REAL multi-conversation chat corpus (DailyDialog mirror) and
#    convert it to WhatsApp-style exports that the production importer reads.
!cd /kaggle/working && python scripts/fetch_public_corpus.py --max-conversations 600
!ls /kaggle/working/sample_corpus/daily_dialog/chats | head -3
""",
    ),
    (
        "code",
        """# 2) Run the experiment: dataset -> conversation split -> LoRA -> evaluation.
#    Every method sees the SAME held-out prompts.
!cd /kaggle/working && python scripts/run_style_experiment.py \\
    --corpus sample_corpus/daily_dialog/chats \\
    --out artifacts/experiments/kaggle \\
    --model qwen3-4b-it-2507 \\
    --methods base,style_profile,style_rag,ngram,lora,style_rag_ft \\
    --train-lora --epochs 2 --lora-rank 16 --learning-rate 1e-4 \\
    --max-train-examples 700 --max-eval-prompts 60 --max-new-tokens 56
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


def build_notebook() -> dict:
    cells = []
    for kind, source in CELLS:
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
            "kaggle": {
                "accelerator": "GPU",
                "enable_gpu": True,
                "gpu_type": "T4",
                "isPrivate": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel_metadata() -> dict:
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "echotwin_lora.ipynb",
        "kernel_type": "notebook",
        "language": "python",
        "is_private": True,
        "enable_gpu": True,
        "gpu_type": "T4",
        "accelerator": "GPU",
        "dataset_sources": [SOURCE_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> None:
    out_dir = Path("notebooks/kaggle")
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / "echotwin_lora.ipynb"
    nb_path.write_text(json.dumps(build_notebook(), indent=1), encoding="utf-8")
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(build_kernel_metadata(), indent=2), encoding="utf-8"
    )
    print(f"wrote {nb_path}")
    print(f"wrote {out_dir / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()

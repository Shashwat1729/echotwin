# Research Roadmap

Implemented only when each item has a valid objective, suitable data,
sufficient compute, and measurable benefit (per the project principles).

## Text

- **Reward-model training / GRPO**: needs a reliable verifiable reward for
  chat style; human pairwise data is the prerequisite.
- **General-data mixing**: small amounts of instruction data to reduce
  style-saturation on tiny personal datasets; measure held-out style +
  contamination before/after.
- **Embedding retrieval**: replace BM25 with multilingual embeddings when
  they beat BM25 on held-out style metrics (BM25 wins on tiny corpora today).
- **Multi-sample evaluation**: k samples per prompt with variance reporting.

## Voice

- **Backend-official adaptation** (VoxCPM LoRA/SFT scripts) as a separate,
  documented workflow with its own data/compute/licensing gates.
- **Independent Hindi evaluators**: Indic ASR + accent-aware speaker
  verification, each with published limitations.
- **Prosody controls**: backend-supported style guidance surfaced with
  capability checks, never emulated.

## Product

- **Gradio → hardened deployment**: auth, rate limits, audit log for shared
  (non-localhost) use. Localhost remains the default.
- **Axolotl/Unsloth export paths**: config translation for users who prefer
  those trainers; our Trainer pipeline stays the reference implementation.
- **On-device inference**: GGUF export + llama.cpp recipes for trained
  adapters (backend-dependent for voice).

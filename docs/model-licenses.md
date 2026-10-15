# Model Licenses

EchoTwin's **code** is Apache-2.0. Everything else carries its own terms —
check before deploying a fine-tune or sharing generated audio.

## Text models (registry, verified 2026-10-15)

| Model | License | Notes |
|---|---|---|
| Qwen3-4B-Instruct-2507 (default) | Apache-2.0 | No license gate. |
| Qwen3.5-4B, Qwen3-1.7B/4B | Apache-2.0 | No license gate. |
| Gemma 4 E4B / Gemma 3n E4B / Gemma 3 | Gemma Terms of Use | Must accept on Hugging Face; responsible-commercial-use terms. |
| Qwen2.5-3B-Instruct (legacy) | Qwen License | Check current Qwen terms. |
| Llama-3.2-3B-Instruct | Llama 3.2 Community License | Requires Meta acceptance. |

## Speech models (verified 2026-10-15)

| Model | License | Notes |
|---|---|---|
| VoxCPM2 (`openbmb/VoxCPM2`) | Apache-2.0 | Model + code. |
| Chatterbox Multilingual | MIT (weights) | Output watermarked (PerTh) by design — good for provenance. |

## Your artifacts

- **Datasets/style profiles**: derived from your own messages — yours.
- **Adapters**: usable under the base model's license terms; Gemma/Llama
  adapters inherit their distribution restrictions.
- **Generated audio**: disclose as AI-generated when shared; watermarked
  backends (Chatterbox) aid provenance, VoxCPM outputs carry our sidecar
  provenance JSON.

When in doubt, keep fine-tunes and voice profiles private and local.

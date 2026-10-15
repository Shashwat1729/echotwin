# Voice Cloning

## Backends (capabilities verified 2026-10-15)

| Backend | Model | Languages | Transcript | Multi-ref | Adaptation | License |
|---|---|---|---|---|---|---|
| `voxcpm` (default) | `openbmb/VoxCPM2` (2B, 48kHz) | 30 incl. hi+en | optional (required for "ultimate" mode) | no | SFT/LoRA via upstream scripts (separate workflow) | Apache-2.0 |
| `chatterbox-multilingual` | ResembleAI | 23 incl. hi+en | no | no | no (reference conditioning only) | MIT (+ watermarked output) |

`echotwin voice backends` prints the live capability matrix. Anything a
backend does not support raises `UnsupportedCapabilityError` — we never
silently substitute.

## Honest limitations

- **Hinglish is not an official language** of any bundled backend. Recording
  plans include optional code-switching tasks labeled as untested; measure
  with the evaluation suite.
- Hindi intelligibility is weaker than English in upstream benchmarks —
  verify before relying on it.
- ~8GB VRAM recommended for VoxCPM2; CPU works but is slow.
- Speaker similarity ≠ identity. Generated audio carries provenance metadata
  and must be disclosed as AI-generated when shared.

## Workflow

```
echotwin voice backends
echotwin voice plan --languages hi,en        # adapts to backend + inventory
echotwin voice scripts --language hi         # real sentences, editable
# record in the UI Voice Studio (browser mic) or: echotwin voice import ./recordings
echotwin voice analyze                       # quality + coverage
echotwin voice recommend                     # next best recording
# select references, grant consent, then:
echotwin voice test
echotwin voice synthesize --text "..."       # user-approved text only
echotwin voice evaluate --audio out.wav <reference...>
```

## Consent

Synthesis requires an explicit `voice_cloning` record: the voice is yours or
you are authorized, and outputs will be disclosed as AI-generated. Revocable
anytime. Ownership of a chat export does **not** imply other participants'
consent.

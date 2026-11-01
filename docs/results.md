# Measured results

Everything on this page was produced by running the code in this repository.
Nothing is estimated, extrapolated, or copied from a paper. Each row names the
command that produced it so you can reproduce it.

Reproduce the environment first:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[voice,train,voice-eval,dev]"
```

---

## 1. Text personalization — real corpus, real model

### Corpus

| Item | Value |
|---|---|
| Source | `ConvLab/dailydialog` (DailyDialog, CC BY 4.0), fetched by `scripts/fetch_public_corpus.py` |
| Shape | 400 real multi-turn dialogues, 3 839 messages, written in WhatsApp export format |
| Author | participant `1` (a consistent author across every dialogue) |
| Conversations | 400 — one per dialogue, so the held-out split is genuinely cross-conversation |
| Licence | CC BY 4.0 (see the dataset card); nothing personal, no real people's messages |

DailyDialog was chosen because it satisfies the two properties a
personalization corpus must have: **one person recurs across many
conversations** (so a style can be learned) and **each dialogue is a separate
conversation** (so a conversation-level split is meaningful). A corpus of
independent one-off utterances cannot produce a leakage-resistant split.

### Pipeline

```bash
python scripts/run_style_experiment.py \
  --corpus sample_corpus/daily_dialog/chats \
  --out artifacts/experiments/local-qwen3 \
  --model qwen3-1.7b --methods base,style_profile,style_rag,ngram
```

Import → target-author selection → context-window dataset → conversation-level
split → BM25 retrieval index (train split only) → per-method generation on
**identical held-out prompts** → style / semantics / fidelity / contamination
metrics.

### Measured (CPU, 24 held-out prompts, Qwen3-1.7B, no GPU)

Full pipeline run `exp-20261021T064125` — 400 conversations / 3 839 messages →
1 931 examples (en 1 546 / hi-Latn 385), split 400/207/301 with the leakage
check passing; 57 context-free prompts skipped; **0 failed generations in any
method**; wall time 3 019 s total on CPU:

| method | char-3gram cosine | length ratio (gen/ref) | distinct ratio | prompt uptake | contamination | failed |
|---|---|---|---|---|---|---|
| `base` (Qwen3-1.7B, no personalization) | 0.798 | **3.08** | 1.00 | 0.248 | clean | 0 |
| `style_profile` (author profile in system prompt) | 0.737 | **0.99** | 1.00 | 0.197 | clean | 0 |
| `style_rag` (keep-retrieval conditioning) | 0.709 | 1.22 | 1.00 | 0.244 | clean | 0 |
| `ngram` (CPU n-gram persona model) | 0.587 | 0.41 | 1.00 | 0.069 | some_overlap | 0 |

Read this honestly:

* **Raw cosine alone would crown the base model** — but that is a rating of
  generic fluent English, not of personalization: the base replies are **3.1×
  longer than the author's** (median 211 vs 60 chars) and ask questions in 4%
  of replies where the author does in 38%. Surface overlap without control of
  length/tone is not style.
* The style-conditioned methods do exactly what their controls target:
  `style_profile` matches reference length almost exactly (0.99) and
  question rate (0.25 vs 0.375), at a modest cosine cost (0.737). Its one
  visible over-correction: it adds emojis in a third of replies where the
  author used none — a profile-tuning defect, recorded as such.
* `style_rag` sits between: shorter-than-profile cosine (0.709) but the **best
  phrase reproduction (0.167)** and base-model uptake (0.244), with lengths
  slightly over reference (1.22).
* The n-gram model is the **weak baseline, and behaves like one**: shortest
  replies (0.41×), no capitalization (distance 0.92), language drift (0.46),
  uptake near zero, and a contamination verdict of `some_overlap` — one of 24
  generations (index 6) was a **verbatim copy of a training target**, which is
  the known failure mode of retrieval-based generation. The zero-GPU option is
  alive (this table was produced with it), it is just not a competitor.
* `distinct ratio 1.00` for **every** method: no method degrades into a canned
  reply across 24 distinct prompts (prompt discrimination — the metric the
  previous audit called the riskiest — holds).
* Prompt uptake stays low across the LoRA-free methods because DailyDialog
  prompts are generic ("How was your day?"); on content prompts the metric is
  the informative one.
* An earlier 3-prompt smoke run (one method, mixed variants) is superseded by
  this table; its numbers are in git history, not repeated here.

LoRA, `style_rag_ft` and DPO arms are the GPU lane (below), the only methods
this table cannot measure locally yet.

### The GPU run (Kaggle)

The same experiment with LoRA training runs on a Kaggle T4:

```bash
kaggle kernels push -p notebooks/kaggle          # notebook + kernel metadata
kaggle kernels output shashwat1729/echotwin-lora-style-transfer -p out/
```

The notebook is `notebooks/kaggle/echotwin_lora.ipynb`; it installs this
package from the private Kaggle dataset `shashwat1729/echotwin-source`, fetches
the corpus itself, trains a LoRA adapter, and evaluates **base vs style_rag vs
LoRA vs style_rag+LoRA** on the same held-out prompts.

#### Result on a real personal chat export (single seed)

Qwen3-1.7B, 2000 training examples, 2 epochs, T4; eval loss 4.63 -> 3.40.
The chat content itself is never published; only these aggregates are.

| Method | style cos | length ratio | distinct | uptake |
|---|---|---|---|---|
| base | 0.754 | 3.41 | 1.00 | 0.303 |
| style_profile (prompt only) | 0.855 | 1.26 | 0.97 | 0.423 |
| style_rag | 0.851 | 1.37 | 0.99 | 0.384 |
| ngram | 0.634 | 0.60 | 0.91 | 0.091 |
| LoRA | 0.768 | 0.94 | 0.82 | 0.086 |
| style_rag + LoRA | 0.773 | 1.06 | 0.93 | 0.114 |

LoRA matches message length best but is more generic; prompt-only style profile
and retrieval score higher on style similarity. Differences are small and come
from one seed, so treat the ranking as indicative, not conclusive.

---

## 2. Voice — speaker similarity, measured on real human speech

The evaluator is the part of the voice pipeline that can be wrong in a way that
*looks* right, so it was measured against real recordings before being trusted.

### Test material (all public, all human)

| Clip set | Source | Speakers | Licence |
|---|---|---|---|
| `sample_audio/hi-speakers/` | FLEURS `hi_in` validation shard | 2 (guaranteed distinct by `gender` column) | CC BY 4.0 |
| `sample_audio/commons-hindi/` | Wikimedia Commons — Lingua Libre Hindi words | 1 (Q1568, "Bhuvanaalla") | CC BY-SA 4.0 |
| `sample_audio/librispeech/` | LibriSpeech `validation` dummy split | 2 (minutes 1089, 1188) | CC BY 4.0 |

### Evaluator comparison (identical clip pairs, same-language only)

| evaluator | same speaker | different speaker | verdict |
|---|---|---|---|
| WavLM x-vector (`microsoft/wavlm-base-plus-sv`) | 0.973 | 0.973 | **does not separate** |
| ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | 0.604 | −0.044 | **separates cleanly** |

This is the single most important measurement in the project. The x-vector
model — the one the ClonEval / Seed-TTS papers use with a fixed 0.7 threshold —
gave a *different* speaker the same 0.973 as the same speaker on read speech.
Any pipeline trusting that threshold would have accepted a clone of the wrong
person.

**Consequences, both implemented:**

1. ECAPA-TDNN is the **primary** evaluator; the x-vector is a fallback with the
   0.7 threshold, and the two are never mixed in one report.
2. A single cosine threshold cannot work for read speech, so the evaluator
   **calibrates against the user's own references**: same-speaker pairs from
   the user's recordings form the baseline, and a clone is judged against that
   baseline rather than a universal constant.

| case | raw score | calibrated | verdict |
|---|---|---|---|
| clone vs same-speaker references | 0.604 | 0.98 | SAME |
| clone vs different-speaker references | −0.044 | −0.07 | DIFFERENT |

### Known limits (documented, not hidden)

* Both embedders are **accent-biased**; an Indian-accent speaker compared with
  a different-accent reference is systematically penalised. Do not use a
  cross-accent pair as a same-speaker baseline.
* ECAPA is trained on VoxCeleb (English conversational speech). Performance on
  read Hindi is weaker than on conversational English, which is exactly why the
  calibration is against *your own* clips.
* Speaker similarity is **not** proof of identity, consent, or authorization.
  It is a similarity measurement, nothing more.

---

## 3. N-gram persona model — decoding work, measured

| change | before | after |
|---|---|---|
| length fidelity (generated / author words) | 1.76 | **1.06** |
| style cosine vs held-out references | 0.355 | **0.397** |
| verbatim copies of training text | 0.25 | **0.00** |
| distinct generations for distinct prompts | 6/8 identical | 8/8 distinct |

The fixes behind those numbers are all in `src/echotwin/models/ngram.py`:
EOS is actually learned, probabilities sum to exactly 1.0, the copy penalty
keeps the prompt conditioning, the sampled length is a *floor* (not a ceiling),
and the EOS probability ramps with position so replies stop where the author
stops.

---

## 4. Honest statement of what is **not** measured

* **No clone was ever synthesised.** No backend weights are installed
  (`pip install voxcpm` pulls ~4 GB), so there is no end-to-end
  "record → clone → score" number. Everything voice-side above validates the
  *evaluator* and the *analysis* on real speech; the synthesis path is
  exercised by tests with a stub backend only.
* **The LoRA comparison is queued, not finished.** The Kaggle kernel is the
  authoritative base-vs-LoRA number; until it completes, the only measured
  text numbers are the CPU ones above.
* DailyDialog is **scripted dialogue** (written for a corpus), not real
  messaging. It has the right *shape* for training and evaluation, but the
  style statistics of a real person will differ.
* No numbers here come from a paper, a blog, or a marketing page. Where the
  literature informed a *design* (PBench axes, the ClonEval protocol, the
  WavLM-SV reference implementation), the design is cited in the code and the
  measurement is our own.

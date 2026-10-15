# Evaluation

## Principle

Every method is scored on the **same frozen held-out prompts**. A report
always contains three parts: style metrics, contamination, and sample-size
honesty. High style similarity **with** high train overlap = memorization.

## Style metrics (surface similarity, not meaning)

- Character 3-gram cosine (reference vs generated corpora).
- Punctuation / capitalization distribution distances.
- Emoji rate (ref vs gen), question rate, length statistics + mean ratio.
- Abbreviation overlap, top-phrase reproduction (labeled *stylometric
  reproduction*, never proof of generalization).
- Language-distribution distance (approximate for Hinglish).

## Contamination (memorization check)

Exact normalized matches + token-6-gram containment against **training**
targets. Verdicts: `clean` / `some_overlap` / `high_overlap`, with flagged
example indices (diagnoses only, no full text dumps).

## Pairwise blind comparison

`echotwin evaluate pairs --a METHOD --b METHOD` aligns two runs on identical
prompts. Human reviewers score style similarity, naturalness, context fit,
and language authenticity (stored as JSONL). `--judge` uses a local model as
judge — results are labeled model-based and never mixed with human scores.

## Voice evaluation

Latency and real-time factor are always measured. Speaker similarity needs an
independent embedding evaluator; WER needs ASR. Missing evaluators yield
`blocked` with install instructions — never invented numbers. Hardware,
backend revision, sample counts, and recording conditions are recorded with
every result.

## Reading results

- n < 30 prompts: indicative only (the report says so).
- One sample per prompt: no sampling variance; re-run with seeds for spread.
- Language detection is approximate for code-mixed Hinglish.
- Never equate speaker similarity with identity, or phrase overlap with
  understanding.

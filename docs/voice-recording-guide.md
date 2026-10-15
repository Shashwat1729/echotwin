# Voice Recording Guide

## Before you start

- Quiet room, consistent mic distance, your normal voice and pace.
- No background music, no overlapping speakers, no aggressive denoisers.
- One consistent microphone and room if possible.
- Record each language naturally — do not fake accents between Hindi and English.

## The studio flow

1. Pick backend + languages → generate a plan (`echotwin voice plan`).
2. Record the **mandatory** clean references first (short, clean, single speaker).
3. Add recommended coverage (statements, questions, numbers, phonetic gaps).
4. The plan tells you to **test the voice** once the backend minimum is met —
   more recording has diminishing returns after that.

## Quality feedback (automated, approximate)

- Too quiet / clipped / long silence / possibly truncated / low SNR estimate.
- Thresholds are configurable (`configs/voice.yaml`). A quiet natural voice
  is never rejected for not matching an assumed loudness profile.

## Transcripts

Type the transcript yourself (best), or configure an ASR provider explicitly —
external services are never called automatically. ASR is **not** ground truth
for Hindi, Romanized Hindi, Hinglish, or names; mismatches are flagged as
*likely* categories and you can always correct the text. The recording is
never altered to match text.

## Reference selection

One best clip per language (unless the backend supports several). Sources are
never concatenated automatically or modified; derived files are separate
artifacts with their own metadata.

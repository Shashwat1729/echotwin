# Hindi–English Pronunciation Coverage

## What the analyzer actually does

Coverage is **text-derived and grapheme-level**, never acoustic phoneme
recognition:

- **Devanagari**: exact grapheme classes from Unicode — dental vs retroflex,
  aspirated vs unaspirated, voiced vs voiceless, nasals, short/long vowels
  and matras, halant conjuncts (clusters). The *mapping* is exact; whether
  you *pronounced* each class correctly is verified only by human listening
  or ASR transcripts.
- **English**: approximate letter classes — vowel letters/digraphs,
  fricatives, nasals, stops, consonant-cluster runs. A character count is
  **not** a phonetic analysis and is never presented as one.
- Unknown classes are reported as `unknown`, never guessed.

## Confidence labels

- `grapheme_exact_text_only` — Devanagari classes.
- `approximate_grapheme` — English letter classes, prosody proxies.
- `unknown` — undeterminable or out of requested languages.

## The script library's job

Starter sentences target dental/retroflex contrast, aspiration, nasals and
vowel length (Hindi); vowel contrasts, clusters, and fricatives (English);
plus numbers, dates, questions, acknowledgements, emotions (calm, no acting),
short voice-note style, and long paragraphs. Everything is editable; skip
what does not match your speech. Hinglish scripts exercise natural switching
with Latin script.

## Goal

Sufficient clean, representative speech for the selected backend — not
maximum recordings. The adaptive plan stops asking once the backend minimum
is met and points you at testing instead.

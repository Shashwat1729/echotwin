# Data Preparation

## Exporting from WhatsApp

- Android: open the chat → ⋮ → More → Export chat (without media is fine).
- iOS: open the chat → contact/group name → Export Chat.
- A `.zip` may contain **many** `_chat.txt` files; EchoTwin imports every
  chat, each as its own conversation (this is what makes held-out splits real).

## What the parser handles

Android/iOS headers, bracketed and unbracketed lines, 12h/24h clocks,
`.`/`/`/`-` separators, DD/MM vs MM/DD (configurable `dayfirst`),
multiline messages, emoji, non-Latin names, group events, attachment
placeholders, deleted/edited markers, truncated files (reported, never fatal).

## What is preserved — and why

Original spelling, capitalization, punctuation, emoji, and grammar are **kept
verbatim**: they are the training signal for style. The only things removed
are structural (timestamps stay as metadata, placeholders stay flagged).

## Target-author selection

The most active participant is **not** assumed to be you (group exports!).
`echotwin participants` shows counts, date ranges, lengths, and language
mix; `echotwin select-author "Name"` records your choice explicitly.

## Dataset construction

Each of your messages becomes a training example with its preceding messages
as context (roles: others → `user`, you → `assistant`). Consecutive messages
are **not** assumed to be reply pairs — examples are labeled `natural_reply`,
`natural_statement`, or `self_continuation`. Large time gaps split segments.
Context is truncated oldest-first within a character budget; targets keep
their exact text.

## Splits and leakage

Splits are by conversation (or chronological), never random adjacent-message
scattering. Overlap across splits is a hard error. One conversation means no
held-out set — the tooling warns and baselines refuse to run rather than
evaluate on training data.

## PII

`ECHOTWIN_REDACT_PII=true` enables pattern-based redaction (emails, phones,
URLs, handles) in processed views. Findings are counts only, never values.

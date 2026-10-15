# Privacy and Security

## Principles

- **Local-first.** Everything lives under `ECHOTWIN_HOME`. No telemetry, no
  external calls without explicit configuration.
- **Explicit consent twice**: once for processing your chat export, once for
  voice cloning (yours or explicitly authorized, with AI-disclosure promise).
- **No impersonation tooling.** No voice-auth bypasses, no provenance hiding.

## Data handling

- Raw message text never enters logs (`log_raw_messages` defaults off and
  must stay off).
- Reports carry counts and pseudonymous ids, never phone numbers or raw
  content. PII redaction (emails, phones, URLs, handles) is configurable;
  findings report counts only.
- Uploads: size-limited, extension-validated, sandboxed to configured
  directories; zip entries screened for traversal.
- API binds to localhost by default; CORS allows only local origins.

## Retention and deletion

- Default retention: keep until you delete (`retention_days: 0`).
- `echotwin privacy export` bundles your state, profiles, and datasets.
- `echotwin privacy delete --scope ... --yes` removes artifacts, profiles,
  datasets, or all local state. Consent records are kept as an audit trail.
- **Documented limit**: deleting files does not guarantee removal of learned
  information from trained adapters — retrain from sanitized data.

## Consent of others

Owning a chat export does not establish other participants' consent. Train on
group data only with appropriate permission, and never publish identifiable
content about others. The public repository contains zero real conversations
or voice samples — only synthetic demo data.

## Threat model

See `SECURITY.md` for reporting and scope. Known residual risks: model
weights may memorize training text (measured by contamination checks, not
prevented); automated audio-quality/SNR estimates are approximate; ASR
transcripts are unreliable for Hindi/Hinglish/names.

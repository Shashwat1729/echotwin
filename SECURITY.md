# Security Policy

EchoTwin processes sensitive personal data (message history, voice
recordings). Security reports are taken seriously.

## Reporting a vulnerability

**Do not open a public issue.** Email the maintainer (see the GitHub profile)
with:

- A description of the vulnerability and its impact.
- Steps to reproduce (synthetic data only — never send real private chats or
  voice recordings).
- Affected version/commit.

Expect an acknowledgement within 7 days. We will coordinate a fix and credit
you unless you prefer anonymity.

## Scope

In scope: path traversal, unsafe uploads, PII leakage into logs/reports,
consent bypasses, insecure defaults (bind address, CORS), dependency
vulnerabilities in bundled code.

Out of scope: social engineering, physical access, vulnerabilities in
third-party base/speech models (report those upstream).

## Threat model (summary)

- Local-first: all data stays under `ECHOTWIN_HOME` unless explicitly
  configured otherwise. The API binds to localhost by default.
- Uploads are size-limited, extension-validated, and sandboxed to configured
  directories; zip entries are screened for traversal.
- Raw message text is never written to logs; PII redaction is configurable.
- Voice synthesis requires an explicit `voice_cloning` consent record.
- Deleting artifacts does not guarantee removal of learned information from
  trained adapters — retrain from sanitized data. See
  `docs/privacy-and-security.md`.

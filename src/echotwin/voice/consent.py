"""Voice consent: explicit authorization gates before any voice-model use.

Two separate gates:

* ``whatsapp_import`` — the user consents to *their own* chat export being processed.
* ``voice_cloning``    — the user confirms in their own words that they own (or are
  authorized to use) the voice being cloned and that outputs will be disclosed as
  AI-generated when shared.

Owning a chat export does NOT establish other participants' consent; training on
group data must respect the ethics documented in docs/privacy-and-security.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from ..storage.db import get_consent, record_consent

SCOPE_WHATSAPP = "whatsapp_import"
SCOPE_VOICE = "voice_cloning"
KNOWN_SCOPES = (SCOPE_WHATSAPP, SCOPE_VOICE)

VOICE_CONSENT_STATEMENT = (
    "I confirm that the voice I am cloning is my own, or that I am explicitly "
    "authorized by the speaker to clone it, and that any generated speech will be "
    "disclosed as AI-generated when shared."
)


class ConsentRecord(BaseModel):
    scope: str
    granted: bool
    granted_at: str | None = None
    details: dict = Field(default_factory=dict)


def _check_scope(scope: str) -> str:
    """Reject unknown scopes.

    A typo in a scope name would otherwise create a consent record that
    nothing ever checks, giving a false sense of coverage.
    """
    if scope not in KNOWN_SCOPES:
        raise ValueError(
            f"Unknown consent scope {scope!r}; known scopes: {list(KNOWN_SCOPES)}"
        )
    return scope


def grant_consent(scope: str, details: dict) -> ConsentRecord:
    _check_scope(scope)
    record_consent(scope, True, details)
    rec = get_consent(scope)
    return ConsentRecord(
        scope=scope,
        granted=rec["granted"],
        granted_at=rec.get("granted_at"),
        details=rec.get("details", {}),
    )


def check_consent(scope: str) -> ConsentRecord:
    rec = get_consent(scope)
    return ConsentRecord(
        scope=scope,
        granted=bool(rec.get("granted")),
        granted_at=rec.get("granted_at"),
        details=rec.get("details", {}),
    )


def require_voice_consent() -> None:
    from ..exceptions import ConsentRequiredError

    rec = check_consent(SCOPE_VOICE)
    if not rec.granted:
        raise ConsentRequiredError(
            "Voice cloning requires explicit consent: confirm that the voice is "
            "your own (or that you have the speaker's authorization). See "
            "docs/privacy-and-security.md."
        )


def consent_prompt_text() -> str:
    return VOICE_CONSENT_STATEMENT


def revoke_consent(scope: str) -> None:
    _check_scope(scope)
    record_consent(scope, False, {"revoked_at": datetime.now(UTC).isoformat()})

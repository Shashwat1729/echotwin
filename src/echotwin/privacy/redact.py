"""Configurable PII redaction.

Only pattern-based detection is used (no network, no external services). Findings
are reported as counts plus stable labels — never the matched values — so logs
and reports cannot leak the PII they describe.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
# Phone: optional +country, 9-15 digits with spaces/dashes/dots/parens
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\(\d{1,4}\)[\s-]?)?\d(?:[\s.-]?\d){8,14}(?!\d)"
)
CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
AADHAAR_LIKE_RE = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)")
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.]{2,30}")

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": EMAIL_RE,
    "url": URL_RE,
    "phone": PHONE_RE,
    "credit_card": CREDIT_CARD_RE,
    "aadhaar_like": AADHAAR_LIKE_RE,
    "handle": HANDLE_RE,
}

PLACEHOLDER = {
    "email": "[EMAIL]",
    "url": "[URL]",
    "phone": "[PHONE]",
    "credit_card": "[CARD]",
    "aadhaar_like": "[ID]",
    "handle": "[HANDLE]",
}


class RedactionConfig(BaseModel):
    enabled: bool = False
    categories: list[str] = Field(default_factory=lambda: list(PATTERNS))
    replacement_style: str = Field(default="placeholder", description="placeholder | mask | drop")


class RedactionReport(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict, description="category -> match COUNT only")
    redacted: bool = False


def redact_text(text: str, cfg: RedactionConfig | None = None) -> tuple[str, RedactionReport]:
    """Return (possibly redacted text, report). Redaction is conservative:
    phone/card patterns only fire on plausible digit runs."""
    cfg = cfg or RedactionConfig()
    if not cfg.enabled:
        return text, RedactionReport(redacted=False)
    out = text
    counts: dict[str, int] = {}
    for cat in cfg.categories:
        rx = PATTERNS.get(cat)
        if rx is None:
            continue
        found = rx.findall(out)
        if not found:
            continue
        counts[cat] = counts.get(cat, 0) + len(found)
        if cfg.replacement_style == "drop":
            out = rx.sub(" ", out)
        else:
            repl = PLACEHOLDER[cat] if cfg.replacement_style == "placeholder" else "█"
            out = rx.sub(repl, out)
    return out, RedactionReport(counts=counts, redacted=True)


def redact_messages(messages: list, cfg: RedactionConfig | None = None):
    """Redact in place-ish: returns new Message objects with text redacted."""
    cfg = cfg or RedactionConfig()
    if not cfg.enabled:
        return messages
    out = []
    for m in messages:
        new_text, _ = redact_text(m.text, cfg)
        mc = m.model_copy(update={"text": new_text})
        out.append(mc)
    return out

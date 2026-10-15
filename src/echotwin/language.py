"""Configurable language identification.

Default is a deterministic heuristic:

* Devanagari script present            -> "hi"
* Devanagari + Latin mixed             -> "hi-en"
* Latin text with Romanized-Hindi markers -> "hi-Latn"
* otherwise                            -> "en"
* empty                                -> "unknown"

Optionally ``langdetect`` (the ``[lang]`` extra) can be used, but it is NOT
deterministic across versions and is disabled by default. The heuristic is
script/heuristic based and documented as approximate; it never guesses phonetics.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum


class Language(StrEnum):
    """Language labels used across ingestion, dataset, and voice modules."""

    HI_DEVANAGARI = "hi"
    EN = "en"
    HI_EN = "hi-en"
    HI_LATN = "hi-Latn"
    UNKNOWN = "unknown"


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F']+")

# Common Romanized-Hindi function words / particles (kept finite and explicit).
_ROMANIZED_HI_MARKERS = {
    "hai",
    "hain",
    "tha",
    "thi",
    "the",
    "ho",
    "hoga",
    "hogi",
    "honge",
    "kya",
    "kyun",
    "kyu",
    "kaise",
    "kahan",
    "kab",
    "kitna",
    "kitne",
    "nahi",
    "nahin",
    "na",
    "haan",
    "ha",
    "ji",
    "mein",
    "main",
    "mai",
    "tu",
    "tum",
    "aap",
    "aapka",
    "aapki",
    "tera",
    "teri",
    "mera",
    "meri",
    "tumhe",
    "tumhein",
    "mujhe",
    "mujhko",
    "unhe",
    "isko",
    "usko",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "par",
    "pe",
    "bhi",
    "toh",
    "to",
    "hi",
    "aur",
    "ya",
    "lekin",
    "magar",
    "agar",
    "phir",
    "abhi",
    "kabhi",
    "kuch",
    "koi",
    "sab",
    "sabhi",
    "bas",
    "matlab",
    "waise",
    "vaisa",
    "aisa",
    "yaar",
    "bhai",
    "boss",
    "arre",
    "acha",
    "accha",
    "achha",
    "theek",
    "thik",
    "karo",
    "karna",
    "karte",
    "karti",
    "kiya",
    "kiye",
    "kar",
    "karke",
    "kraya",
    "raha",
    "rahi",
    "rahe",
    "gaya",
    "gayi",
    "gaye",
    "liya",
    "diya",
    "hua",
    "hui",
    "chahiye",
    "chaiye",
    "sakta",
    "sakti",
    "sakte",
    "mila",
    "mili",
    "mile",
    "bata",
    "batao",
    "dekh",
    "dekho",
    "dekha",
    "suno",
    "suna",
    "chalo",
    "aaja",
    "aao",
    "dena",
    "lena",
    "rehna",
    "khana",
    "chal",
    # Numbers, time, and everyday words: Hinglish chats are full of these,
    # and they are strong Romanized-Hindi signals (Latin script first).
    "ek",
    "do",
    "teen",
    "chaar",
    "paanch",
    "chhah",
    "saat",
    "aath",
    "nau",
    "das",
    "gyaarah",
    "baarah",
    "sau",
    "hazaar",
    "subah",
    "dopahar",
    "shaam",
    "raat",
    "din",
    "kal",
    "aaj",
    "waqt",
    "ghar",
    "kaam",
    "chai",
    "dost",
    "mausam",
    "khush",
    "bada",
    "chhota",
    "naya",
    "purana",
    "bura",
    "jaldi",
    "bahar",
    "andar",
    "upar",
    "neeche",
    "paas",
    "dur",
    "jawab",
    "sawal",
    "padha",
    "tumne",
    "scene",
    "baje",
    "chhutti",
    "karunga",
    "karenge",
    "milte",
    "rukna",
}

_ROMANIZED_HI_THRESHOLD = 0.12
_MIXED_SCRIPT_THRESHOLD = 0.10  # share of letters


def _script_ratios(text: str) -> tuple[float, float]:
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return 0.0, 0.0
    dev = len(_DEVANAGARI_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    return dev / letters, lat / letters


def detect_language(text: str) -> Language:
    """Deterministic script + Romanized-Hindi heuristic. Approximate by design."""
    if not text or not text.strip():
        return Language.UNKNOWN
    dev_ratio, lat_ratio = _script_ratios(text)
    has_dev = dev_ratio >= 0.02
    has_lat = lat_ratio >= 0.02
    if has_dev and has_lat and dev_ratio >= _MIXED_SCRIPT_THRESHOLD:
        return Language.HI_EN
    if has_dev:
        return Language.HI_DEVANAGARI
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return Language.UNKNOWN
    hits = sum(
        1
        for w in words
        if w in _ROMANIZED_HI_MARKERS or w in {m.strip() for m in _ROMANIZED_HI_MARKERS}
    )
    if hits and hits / len(words) >= _ROMANIZED_HI_THRESHOLD:
        return Language.HI_LATN
    return Language.EN


def detect_language_with_confidence(text: str) -> tuple[Language, float, str]:
    """Return (language, confidence 0-1, detector name).

    Confidence reflects marker strength for the heuristic, not phonetic certainty.
    """
    if not text or not text.strip():
        return Language.UNKNOWN, 0.0, "heuristic-empty"
    dev_ratio, lat_ratio = _script_ratios(text)
    if dev_ratio >= _MIXED_SCRIPT_THRESHOLD and lat_ratio >= _MIXED_SCRIPT_THRESHOLD:
        return Language.HI_EN, min(0.99, 0.6 + dev_ratio + lat_ratio), "heuristic-script-mix"
    if dev_ratio >= 0.02:
        return Language.HI_DEVANAGARI, min(0.99, 0.5 + dev_ratio), "heuristic-script"
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if words:
        hits = sum(1 for w in words if w in _ROMANIZED_HI_MARKERS)
        frac = hits / len(words)
        if frac >= _ROMANIZED_HI_THRESHOLD:
            return Language.HI_LATN, min(0.95, 0.4 + frac), "heuristic-romanized-marker"
    return Language.EN, 0.5 if words else 0.0, "heuristic-default"


def script_notes() -> dict[str, str]:
    """Documented limitations of the heuristic, surfaced in reports."""
    return {
        "method": "script ratios + Romanized-Hindi marker lexicon (heuristic)",
        "known_limits": (
            "Cannot distinguish fluent Hindi written in Latin script from English without "
            "enough markers; cannot identify other languages (Urdu, Marathi in Devanagari, "
            "etc.). Treat hi-Latn vs en boundaries as approximate."
        ),
    }


def unicode_category_sample(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace():
            continue
        name = unicodedata.category(ch)
        counts[name] = counts.get(name, 0) + 1
    return counts

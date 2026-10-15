"""Script-coverage analysis (documented, approximate phonetic analysis).

Method (documented, not mystical):

* **Devanagari**: grapheme->class mapping is *exact at the grapheme level* —
  dental त थ द ध न vs retroflex ट ठ ड ढ ण vs aspirated/unaspirated etc. What we
  CANNOT verify from text is whether the speaker actually pronounced the class
  correctly (sandhi, accent, slips). So each category reports ``grapheme_coverage:
  true/false`` plus ``confidence="grapheme_exact_text_only"``.
* **English**: letter-class approximation (vowel letters, fricative letters,
  consonant clusters). Explicitly ``confidence="approximate_grapheme"`` — a
  character count is NOT a phonetic analysis and is never presented as one.
* If a class cannot be determined, it is reported as ``unknown`` — never guessed.
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, Field

from ..language import _ROMANIZED_HI_MARKERS
from ..voice.script_library import ScriptCategory, ScriptEntry

# --- Devanagari grapheme classes (Unicode block, documented mapping) ---------
DEVA_DENTAL = set("तथदधन")
DEVA_RETROFLEX = set("टठडढण")
DEVA_ASPIRATED = set("खगघछझठढथधफभ")
DEVA_VOICELESS = set("कखचछटठतथपफ")
DEVA_VOICED = set("गघजझडढदधबभ")
DEVA_NASALS = set("ङञणनम")
DEVA_FRICATIVES = set("शषसह")
DEVA_SEMIVOWELS = set("यरलव")
DEVA_SHORT_VOWELS = set("अइउएओ")
DEVA_LONG_VOWELS = set("आईऊएऐओ")
DEVA_LONG_MATRAS = set("ीूेैोौाृ")
DEVA_SHORT_MATRAS = set("िु")
DEVA_NASALIZATION = set("ंँ")
DEVA_HALANT = "्"

EN_VOWEL_LETTERS = set("aeiouy")
EN_FRICATIVE_LETTERS = set("fvsz")
EN_NASAL_LETTERS = set("mn")
EN_STOP_LETTERS = set("pbtdkg")
EN_AFFRICATE_DIGRAPHS = {"ch", "sh", "th", "ph", "wh"}

_DEVA_CHAR_RE = re.compile(r"[\u0900-\u097F]")
_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F']+")

_CATEGORY_LABELS = {
    "hi_dental": "Hindi dental consonants (त थ द ध न)",
    "hi_retroflex": "Hindi retroflex consonants (ट ठ ड ढ ण)",
    "hi_aspirated": "Hindi aspirated consonants (ख घ छ झ ठ ढ थ ध फ भ)",
    "hi_unaspirated": "Hindi unaspirated consonants (क ग च ज ट ड त द प ब)",
    "hi_voiced": "Hindi voiced stops",
    "hi_voiceless": "Hindi voiceless stops",
    "hi_nasals": "Hindi nasals (ङ ञ ण न म)",
    "hi_long_vowels": "Hindi long vowels/matras",
    "hi_short_vowels": "Hindi short vowels/matras",
    "hi_clusters": "Hindi consonant clusters (halant conjuncts)",
    "en_vowels": "English vowel letters/digraphs",
    "en_fricatives": "English fricatives (f v s z + th sh)",
    "en_nasals": "English nasals (m n ng)",
    "en_clusters": "English consonant clusters (CC sequences)",
    "en_stops": "English stop letters (p b t d k g)",
    "prosody_question": "Question prosody (sentences ending in '?')",
    "prosody_statement": "Statement prosody (sentences ending in '.'/'!')",
}


class CategoryCoverage(BaseModel):
    count: int = 0
    covered: bool = False
    confidence: str = Field(description="grapheme_exact_text_only | approximate_grapheme | unknown")
    note: str = ""


class CoverageReport(BaseModel):
    coverage_version: str = "1.0"
    n_texts_analyzed: int = 0
    categories: dict[str, CategoryCoverage] = Field(default_factory=dict)
    languages_present: dict[str, int] = Field(default_factory=dict)
    scripts_present: dict[str, int] = Field(default_factory=dict)
    method_notes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def _analyze_devanagari(text: str, acc: Counter) -> None:
    for ch in text:
        if ch in DEVA_DENTAL:
            acc["hi_dental"] += 1
        if ch in DEVA_RETROFLEX:
            acc["hi_retroflex"] += 1
        if ch in DEVA_ASPIRATED:
            acc["hi_aspirated"] += 1
        if ch in DEVA_VOICELESS and ch not in DEVA_ASPIRATED:
            acc["hi_unaspirated"] += 1
        if ch in DEVA_VOICED and ch not in DEVA_ASPIRATED:
            acc["hi_voiced"] += 1
        if ch in DEVA_VOICELESS:
            acc["hi_voiceless"] += 1
        if ch in DEVA_NASALS:
            acc["hi_nasals"] += 1
        if ch in DEVA_LONG_VOWELS or ch in DEVA_LONG_MATRAS:
            acc["hi_long_vowels"] += 1
        if ch in DEVA_SHORT_VOWELS or ch in DEVA_SHORT_MATRAS:
            acc["hi_short_vowels"] += 1
        if ch in DEVA_NASALIZATION:
            acc["hi_nasals"] += 1
        if ch == DEVA_HALANT:
            acc["hi_clusters"] += 1


def _analyze_english(text: str, acc: Counter) -> None:
    words = _WORD_RE.findall(text.lower())
    for w in words:
        # strip devanagari (handled elsewhere)
        if any("\u0900" <= c <= "\u097f" for c in w):
            continue
        letters = [c for c in w if c.isalpha()]
        if not letters:
            continue
        for c in letters:
            if c in EN_VOWEL_LETTERS:
                acc["en_vowels"] += 1
            elif c in EN_FRICATIVE_LETTERS:
                acc["en_fricatives"] += 1
            elif c in EN_NASAL_LETTERS:
                acc["en_nasals"] += 1
            elif c in EN_STOP_LETTERS:
                acc["en_stops"] += 1
        joined = "".join(letters)
        if any(d in joined for d in EN_AFFRICATE_DIGRAPHS):
            acc["en_fricatives"] += 1
        # consonant clusters: 2+ adjacent non-vowel letters
        run = 0
        for c in letters:
            if c not in EN_VOWEL_LETTERS:
                run += 1
                if run >= 2:
                    acc["en_clusters"] += 1
            else:
                run = 0


# --- Romanized Hindi (Hinglish in Latin script): approximate by design ------
# Dental vs retroflex is NOT distinguishable in Latin script, so those two
# categories are deliberately never touched here (stay unknown unless
# Devanagari evidence exists). Everything below is confidence
# "approximate_romanized": useful for planning, never phoneme-grade.
_ROM_ASPIRATED_DIGRAPHS = ("kh", "gh", "chh", "jh", "th", "dh", "ph", "bh")
_ROM_LONG_VOWELS = ("aa", "ee", "oo", "ai", "au", "ou")
_ROM_VOICED_STOPS = ("g", "j", "d", "b")
_ROM_VOICELESS_STOPS = ("k", "c", "t", "p")
_ROM_MARKER_FRACTION = 0.12

# English function words excluded from romanized analysis: in Hinglish text
# the content words carry the Hindi phonetics, the function words don't.
_ROM_EN_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "that",
        "this",
        "it",
        "its",
        "as",
        "at",
        "by",
        "from",
        "you",
        "we",
        "they",
        "he",
        "she",
        "him",
        "her",
        "them",
        "will",
        "would",
        "can",
        "could",
        "let",
        "need",
        "want",
        "time",
        "plan",
        "step",
        "next",
        "first",
        "option",
        "options",
        "message",
        "call",
        "wait",
        "sense",
        "makes",
        "compare",
        "rush",
        "check",
        "minute",
        "reply",
        "meeting",
        "meet",
        "today",
        "now",
        "hello",
        "hi",
        "no",
        "not",
        "so",
        "that",
        "there",
        "here",
        "what",
        "when",
        "where",
        "why",
        "how",
        "my",
        "your",
        "our",
        "their",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "get",
        "got",
        "make",
        "made",
        "take",
        "like",
        "just",
        "only",
        "also",
        "very",
        "than",
        "then",
        "too",
        "over",
        "under",
        "between",
        "actually",
        "honestly",
        "basically",
    ]
)


def _is_devanagari_word(word: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x097F for c in word)


def _romanized_hindi_words(text: str) -> list[str]:
    """Latin words likely carrying Hindi phonetics in a Hinglish text."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    words = [w for w in words if w.isalpha() and not _is_devanagari_word(w)]
    if not words:
        return []
    hits = sum(1 for w in words if w in _ROMANIZED_HI_MARKERS)
    if hits / len(words) < _ROM_MARKER_FRACTION:
        return []
    return [w for w in words if w not in _ROM_EN_STOPWORDS]


def _analyze_romanized(words: list[str], acc: Counter) -> None:
    for lowered in words:
        if any(d in lowered for d in _ROM_ASPIRATED_DIGRAPHS):
            acc["hi_aspirated"] += 1
        else:
            stops = [c for c in lowered if c in "kgjcctdpb"]
            if stops:
                acc["hi_unaspirated"] += 1
        plain = lowered
        for d in _ROM_ASPIRATED_DIGRAPHS:
            plain = plain.replace(d, "")
        if any(c in plain for c in _ROM_VOICED_STOPS):
            acc["hi_voiced"] += 1
        if any(c in plain for c in _ROM_VOICELESS_STOPS):
            acc["hi_voiceless"] += 1
        if any(c in lowered for c in "nm"):
            acc["hi_nasals"] += 1
        if any(v in lowered for v in _ROM_LONG_VOWELS):
            acc["hi_long_vowels"] += 1
        if any(c in lowered for c in "aiu"):
            acc["hi_short_vowels"] += 1
        run = 0
        for c in lowered:
            if c.isalpha() and c not in "aeiouy":
                run += 1
                if run >= 2:
                    acc["hi_clusters"] += 1
            else:
                run = 0


def analyze_texts(
    texts: list[str],
    *,
    languages: list[str] | None = None,
) -> CoverageReport:
    """Analyze a list of texts (scripts actually recorded/read)."""
    acc: Counter = Counter()
    roman: Counter = Counter()
    lang_acc: Counter = Counter()
    script_acc: Counter = Counter()
    for text in texts:
        if not text or not text.strip():
            continue
        has_deva = bool(_DEVA_CHAR_RE.search(text))
        script_acc["devanagari" if has_deva else "latin"] += 1
        if has_deva:
            lang_acc["hi"] = lang_acc.get("hi", 0) + 1
        else:
            # crude script-level guess; fine-grained language needs the language module
            lang_acc["latin"] = lang_acc.get("latin", 0) + 1
        _analyze_devanagari(text, acc)
        _analyze_english(text, acc)
        if not has_deva:
            _analyze_romanized(_romanized_hindi_words(text), roman)
        if "?" in text:
            acc["prosody_question"] += 1
        if re.search(r"[.!…]\s*$", text.strip()):
            acc["prosody_statement"] += 1

    categories: dict[str, CategoryCoverage] = {}
    for key, label in _CATEGORY_LABELS.items():
        count = acc.get(key, 0) + roman.get(key, 0)
        if acc.get(key, 0) > 0:
            conf = "grapheme_exact_text_only" if key.startswith("hi_") else "approximate_grapheme"
        elif roman.get(key, 0) > 0:
            conf = "approximate_romanized"
        else:
            conf = (
                "approximate_grapheme" if not key.startswith("hi_") else "grapheme_exact_text_only"
            )
        categories[key] = CategoryCoverage(
            count=count,
            covered=count > 0,
            confidence=conf,
            note=label,
        )
        if languages is not None and not _category_relevant(key, languages):
            categories[key] = CategoryCoverage(
                count=0, covered=False, confidence="unknown", note=f"{label} (not requested)"
            )

    return CoverageReport(
        n_texts_analyzed=len([t for t in texts if t and t.strip()]),
        categories=categories,
        languages_present=dict(lang_acc),
        scripts_present=dict(script_acc),
        method_notes=[
            "Devanagari: exact grapheme classes from Unicode; pronunciation verified only by human/ASR.",
            "English: letter-class approximation; clusters via consonant runs.",
            "Romanized Hindi (Hinglish in Latin script): approximate classes from "
            "aspirated digraphs, nasal/long-vowel spelling and consonant runs, "
            "counted only in marker-rich text minus English function words. "
            "Dental vs retroflex is NOT distinguishable in Latin script and is "
            "never claimed from romanized evidence.",
        ],
        limitations=[
            "This is a TEXT-derived coverage estimate, not acoustic phoneme recognition.",
            "Coverage of a grapheme class says nothing about the quality of the spoken sound.",
            "Run ASR transcript verification on recordings for pronunciation evidence.",
        ],
    )


def _category_relevant(key: str, languages: list[str]) -> bool:
    lowered = [lang.lower() for lang in languages]
    if key.startswith("hi_"):
        return any(lang.startswith("hi") or lang in {"hinglish", "hi-latn"} for lang in lowered)
    if key.startswith("en_"):
        return any(
            lang.startswith("en") or lang in {"hinglish", "hi-latn", "hi-en"} for lang in lowered
        )
    return True


def missing_categories(report: CoverageReport) -> list[str]:
    return [k for k, v in report.categories.items() if not v.covered]


def covered_categories(report: CoverageReport) -> list[str]:
    return [k for k, v in report.categories.items() if v.covered]


# Which script categories address which coverage categories
COVERAGE_TO_SCRIPT_CATEGORIES: dict[str, list[str]] = {
    "hi_dental": [ScriptCategory.PHONETIC_HI_DENTAL_RETROFLEX.value],
    "hi_retroflex": [ScriptCategory.PHONETIC_HI_DENTAL_RETROFLEX.value],
    "hi_aspirated": [ScriptCategory.PHONETIC_HI_ASPIRATION.value],
    "hi_unaspirated": [
        ScriptCategory.PHONETIC_HI_ASPIRATION.value,
        ScriptCategory.CLEAN_REFERENCE.value,
    ],
    "hi_voiced": [ScriptCategory.CLEAN_REFERENCE.value],
    "hi_voiceless": [ScriptCategory.CLEAN_REFERENCE.value],
    "hi_nasals": [ScriptCategory.PHONETIC_HI_NASALS_VOWELS.value],
    "hi_long_vowels": [ScriptCategory.PHONETIC_HI_NASALS_VOWELS.value],
    "hi_short_vowels": [ScriptCategory.PHONETIC_HI_NASALS_VOWELS.value],
    "hi_clusters": [
        ScriptCategory.PHONETIC_HI_CLUSTERS.value,
        ScriptCategory.LONG_PARAGRAPH.value,
    ],
    "en_vowels": [ScriptCategory.PHONETIC_EN_VOWELS.value],
    "en_fricatives": [ScriptCategory.PHONETIC_EN_FRICATIVES.value],
    "en_nasals": [ScriptCategory.PHONETIC_EN_FRICATIVES.value],
    "en_clusters": [ScriptCategory.PHONETIC_EN_CLUSTERS.value],
    "en_stops": [ScriptCategory.CLEAN_REFERENCE.value],
    "prosody_question": [ScriptCategory.QUESTION.value],
    "prosody_statement": [ScriptCategory.STATEMENT.value, ScriptCategory.EXPLANATION.value],
}


def recommend_scripts_for_gaps(
    report: CoverageReport, language: str, *, limit: int = 6
) -> list[ScriptEntry]:
    """Map missing coverage categories back to library scripts of that language."""
    from ..voice.script_library import SCRIPT_LIBRARY

    wanted: list[str] = []
    for cat in missing_categories(report):
        wanted.extend(COVERAGE_TO_SCRIPT_CATEGORIES.get(cat, []))
    seen: set[str] = set()
    out: list[ScriptEntry] = []
    for entry in SCRIPT_LIBRARY:
        if entry.language.value != language:
            continue
        if entry.category.value in wanted and entry.id not in seen:
            seen.add(entry.id)
            out.append(entry)
        if len(out) >= limit:
            break
    return out


def estimate_additional_seconds(entries: list[ScriptEntry]) -> float:
    return round(sum(e.est_seconds for e in entries), 1)

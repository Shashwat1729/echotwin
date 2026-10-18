"""Adaptive recording-plan generator.

Given a backend capability profile, user preferences, the recording inventory and
the coverage report, produce a prioritized, budget-aware plan. The plan is
deterministic (same inputs → same plan) and NEVER invents requirements the
backend does not have:

* Only languages the backend supports are planned (others go to ``skipped``).
* Mandatory tasks cover only the backend's own minimum (it needs at least one
  clean reference clip).
* Hinglish tasks are always optional with an explicit note that code-switching
  is not an official backend language.
* When mandatory coverage is satisfied, the plan recommends TESTING the voice,
  not more recording.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from ..voice.capabilities import VoiceBackendCapabilities
from ..voice.phoneme_coverage import (
    CoverageReport,
    estimate_additional_seconds,
    recommend_scripts_for_gaps,
)
from ..voice.script_library import ScriptEntry, all_scripts

DEFAULT_TARGET_COPIES_PER_LANGUAGE = 1  # clean reference minimum per language


class RecordingPreferences(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["hi", "en"])
    include_hinglish: bool = True
    speaker_profile: str = Field(
        default="bilingual",
        description="native_hindi | indian_english | bilingual | natural_hinglish",
    )
    script_preference: str = Field(default="auto", description="auto | devanagari | latin")
    accent_note: str | None = None
    max_minutes: float = 20.0


class InventoryItem(BaseModel):
    recording_id: str
    language: str
    quality_verdict: str = "ok"  # ok | warning | fail
    duration_seconds: float = 0.0
    category: str = "uncategorized"
    has_transcript: bool = False
    coverage_categories_present: list[str] = Field(default_factory=list)


class RecordingTask(BaseModel):
    id: str
    language: str
    category: str
    title: str
    script_ids: list[str]
    priority: int
    mandatory: bool
    rationale: str
    est_seconds: float
    optional_improvement: bool = Field(
        default=True, description="True unless the backend hard-requires it"
    )
    beyond_time_budget: bool = False
    notes: list[str] = Field(default_factory=list)


class RecordingPlan(BaseModel):
    backend: str
    backend_capabilities: dict
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    tasks: list[RecordingTask] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)
    total_estimated_seconds: float = 0.0
    within_time_budget_seconds: float = 0.0
    backend_minimum_met: bool = False
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def quality_ok(item: InventoryItem) -> bool:
    return item.quality_verdict in {"ok", "warning"}


def _count_ok(inventory: list[InventoryItem], language: str) -> list[InventoryItem]:
    return [i for i in inventory if i.language == language and quality_ok(i)]


def _pick_scripts(language: str, categories: list[str], limit: int) -> list[ScriptEntry]:
    out: list[ScriptEntry] = []
    for entry in all_scripts():
        if entry.language.value != language:
            continue
        if entry.category.value in categories:
            out.append(entry)
        if len(out) >= limit:
            break
    return out


def generate_plan(
    capabilities: VoiceBackendCapabilities | dict,
    prefs: RecordingPreferences,
    inventory: list[InventoryItem],
    coverage: CoverageReport | None = None,
) -> RecordingPlan:
    caps = (
        capabilities
        if isinstance(capabilities, VoiceBackendCapabilities)
        else VoiceBackendCapabilities(**capabilities)
    )
    warnings: list[str] = []
    skipped: list[dict] = []
    tasks: list[RecordingTask] = []
    budget_seconds = prefs.max_minutes * 60.0
    cumulative = 0.0

    def add_task(task: RecordingTask) -> None:
        nonlocal cumulative
        if cumulative + task.est_seconds > budget_seconds and not task.mandatory:
            task.beyond_time_budget = True
        else:
            cumulative += task.est_seconds
        tasks.append(task)

    # 1. Language gating by capability.
    requested = [lang for lang in prefs.languages]
    if prefs.include_hinglish and "hinglish" not in requested:
        requested.append("hinglish")

    def backend_lang_ok(lang: str) -> bool:
        if lang == "hinglish":
            return caps.supports_language("hi") and caps.supports_language("en")
        return caps.supports_language(lang)

    for lang in requested:
        if not backend_lang_ok(lang):
            skipped.append(
                {
                    "language": lang,
                    "reason": "unsupported_language",
                    "detail": f"Backend '{caps.name}' does not list '{lang}'. No tasks planned.",
                }
            )
    planned_langs = [lang for lang in requested if backend_lang_ok(lang)]

    # 2. Essential clean reference per language (the backend hard minimum).
    for lang in planned_langs:
        if lang == "hinglish":
            continue
        have = _count_ok(inventory, lang)
        enougher = [i for i in have if i.duration_seconds >= caps.min_reference_seconds]
        if not enougher:
            scripts = _pick_scripts(lang, ["clean_reference"], 1)
            task = RecordingTask(
                id=f"essential-{lang}",
                language=lang,
                category="clean_reference",
                title=f"Clean {lang.upper()} reference recording",
                script_ids=[s.id for s in scripts],
                priority=1,
                mandatory=True,
                rationale=(
                    f"The backend requires at least one clean reference clip of "
                    f"{caps.min_reference_seconds:.0f}+ seconds in {lang}."
                ),
                est_seconds=estimate_additional_seconds(scripts)
                or caps.recommended_reference_seconds,
                optional_improvement=False,
                notes=(
                    [
                        "Script contains [NAME]/[DATE]/[TIME] placeholders — fill them in "
                        "or ask the system to generate today's date sentence before recording."
                    ]
                    if any(s.contains_placeholders for s in scripts)
                    else []
                ),
            )
            add_task(task)

    # 3. Recommended coverage per language (2nd sample).
    for lang in planned_langs:
        if lang == "hinglish":
            continue
        have = _count_ok(inventory, lang)
        if len(have) < 2:
            scripts = _pick_scripts(lang, ["statement", "question", "numbers"], 3)
            if scripts:
                task = RecordingTask(
                    id=f"coverage-{lang}",
                    language=lang,
                    category="language_coverage",
                    title=f"{lang.upper()} statement/question/number samples",
                    script_ids=[s.id for s in scripts],
                    priority=2,
                    mandatory=False,
                    rationale=(
                        "A second clean sample per language improves reference "
                        "stability, and numbers/questions exercise prosody."
                    ),
                    est_seconds=estimate_additional_seconds(scripts),
                    optional_improvement=True,
                )
                add_task(task)

    # 4. Hinglish (code-switching) — optional, capability-note included.
    if "hinglish" in planned_langs:
        have = _count_ok(inventory, "hinglish")
        if len(have) < 2:
            scripts = _pick_scripts("hinglish", ["code_switch", "conversational"], 2)
            if scripts:
                add_task(
                    RecordingTask(
                        id="coverage-hinglish",
                        language="hinglish",
                        category="code_switch",
                        title="Natural Hinglish code-switching samples",
                        script_ids=[s.id for s in scripts],
                        priority=2,
                        mandatory=False,
                        rationale="You speak mixed Hindi-English; the backend clones timbre "
                        "but has NO official Hinglish language mode.",
                        est_seconds=estimate_additional_seconds(scripts),
                        optional_improvement=True,
                        notes=[
                            "Code-switching quality is untested for this backend; "
                            "test after recording and judge from the evaluation suite.",
                        ],
                    )
                )

    # 5. Phonetic gaps (coverage-driven), optional.
    if coverage is not None:
        for lang in planned_langs:
            gap_scripts = recommend_scripts_for_gaps(coverage, lang, limit=4)
            if not gap_scripts:
                continue
            ids = [s.id for s in gap_scripts]
            if any(t.script_ids == ids for t in tasks):
                continue
            add_task(
                RecordingTask(
                    id=f"phonetic-{lang}",
                    language=lang,
                    category="phonetic_coverage",
                    title=f"Close phonetic gaps for {lang}",
                    script_ids=ids,
                    priority=3,
                    mandatory=False,
                    rationale="These sentences cover missing grapheme-level categories in "
                    "your recordings.",
                    est_seconds=estimate_additional_seconds(gap_scripts),
                    optional_improvement=True,
                )
            )

    # 6. Final recommendation: test if minimum met, else record essentials first.
    mandatory_tasks = [t for t in tasks if t.mandatory]
    # The minimum is only met when there is at least one planned language AND
    # every one of them has a qualifying recording. `all([])` is True, so a
    # backend that supports none of the requested languages used to report
    # "minimum met — go test the voice" for a user with zero recordings.
    _gate_langs = [lang for lang in planned_langs if lang != "hinglish"]
    backend_minimum_met = (
        bool(_gate_langs)
        and not mandatory_tasks
        and all(
            any(
                i.duration_seconds >= caps.min_reference_seconds for i in _count_ok(inventory, lang)
            )
            for lang in _gate_langs
        )
    )
    recommendations: list[str] = []
    if backend_minimum_met:
        add_task(
            RecordingTask(
                id="test-voice",
                language="-",
                category="evaluation",
                title="Test the voice before recording more",
                script_ids=[],
                priority=99,
                mandatory=False,
                rationale="Backend minimum is satisfied; synthesize a test sentence and "
                "evaluate before spending more recording time.",
                est_seconds=30.0,
                optional_improvement=True,
                notes=["Run: echotwin voice test"],
            )
        )
        recommendations.append("Backend minimum met — prioritize testing over recording.")
    elif mandatory_tasks:
        recommendations.append(
            f"Record the mandatory task(s) first: {', '.join(t.id for t in mandatory_tasks)}."
        )
    else:
        recommendations.append("Start with the mandatory task(s).")

    # 7. Warnings for budget.
    if any(t.beyond_time_budget for t in tasks):
        warnings.append(
            f"Some optional tasks exceed your {prefs.max_minutes:.0f}-minute budget and are "
            "marked beyond_time_budget; skip them if time is short."
        )
    if caps.license and "Apache" in caps.license:
        pass
    return RecordingPlan(
        backend=caps.name,
        backend_capabilities=caps.public_dict(),
        tasks=sorted(tasks, key=lambda t: (t.priority, t.id)),
        skipped=skipped,
        total_estimated_seconds=round(sum(t.est_seconds for t in tasks), 1),
        # Mandatory tasks are exempt from the budget, so the raw sum could
        # exceed the budget itself. Report the budgeted time, not more.
        within_time_budget_seconds=round(
            min(
                sum(t.est_seconds for t in tasks if not t.beyond_time_budget),
                budget_seconds,
            ),
            1,
        ),
        backend_minimum_met=backend_minimum_met,
        recommendations=recommendations
        or [
            (
                "Backend minimum met — prioritize testing over recording."
                if backend_minimum_met
                else "Start with the mandatory task(s)."
            )
        ],
        warnings=warnings,
    )

"""Voice package: bilingual recording studio backends, analysis, and evaluation."""

from .audio_ingestion import AudioClip, AudioFormatError, read_audio, validate_recording
from .audio_quality import (
    QualityIssue,
    QualityReport,
    QualityThresholds,
    analyze_audio_file,
    analyze_wave_array,
    summary_line,
)
from .capabilities import VoiceBackendCapabilities
from .consent import (
    ConsentRecord,
    check_consent,
    grant_consent,
    require_voice_consent,
    revoke_consent,
)
from .metadata import (
    RecordingEntry,
    VoiceProfile,
    add_recording,
    delete_profile,
    export_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from .phoneme_coverage import (
    CoverageReport,
    analyze_texts,
    covered_categories,
    missing_categories,
    recommend_scripts_for_gaps,
)
from .recording_plan import (
    InventoryItem,
    RecordingPlan,
    RecordingPreferences,
    RecordingTask,
    generate_plan,
)
from .reference_selection import ReferenceCandidate, SelectionDecision, select_references
from .registry import (
    all_backend_statuses,
    backend_status,
    find_available_backend,
    get_backend,
    registered_backends,
)
from .script_library import (
    SCRIPT_LIBRARY,
    ScriptEntry,
    ScriptLanguage,
    all_scripts,
    get_script,
    render,
    scripts_by_category,
    scripts_by_language,
)
from .synthesis import SynthesisResult, synthesize
from .transcript_validation import TranscriptCheck, verify_transcript

__all__ = [
    "SCRIPT_LIBRARY",
    "AudioClip",
    "AudioFormatError",
    "ConsentRecord",
    "CoverageReport",
    "InventoryItem",
    "QualityIssue",
    "QualityReport",
    "QualityThresholds",
    "RecordingEntry",
    "RecordingPlan",
    "RecordingPreferences",
    "RecordingTask",
    "ReferenceCandidate",
    "ScriptEntry",
    "ScriptLanguage",
    "SelectionDecision",
    "SynthesisResult",
    "TranscriptCheck",
    "VoiceBackendCapabilities",
    "VoiceProfile",
    "add_recording",
    "all_backend_statuses",
    "all_scripts",
    "analyze_audio_file",
    "analyze_texts",
    "analyze_wave_array",
    "backend_status",
    "check_consent",
    "covered_categories",
    "delete_profile",
    "export_profile",
    "find_available_backend",
    "generate_plan",
    "get_backend",
    "get_script",
    "grant_consent",
    "list_profiles",
    "load_profile",
    "missing_categories",
    "read_audio",
    "recommend_scripts_for_gaps",
    "registered_backends",
    "render",
    "require_voice_consent",
    "revoke_consent",
    "save_profile",
    "scripts_by_category",
    "scripts_by_language",
    "select_references",
    "summary_line",
    "synthesize",
    "validate_recording",
    "verify_transcript",
]

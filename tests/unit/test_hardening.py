"""Tests for the hardening pass: streaming uploads, request bounds, stale-job
repair, consent evidence, honest deletion scopes, prompt-uptake, loader and
doctor heuristics.

Every test here pins a behavior change whose absence previously caused a
silent wrong result (a 500 for bad input, a job stuck 'queued' forever, a
consent revocation that erased its own evidence, an upload buffered in RAM).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from echotwin.api.schemas import DatasetPrepareRequest, TrainRequest
from echotwin.evaluation import count_context_free, heldout_prompts
from echotwin.evaluation.baseline import compare_runs
from echotwin.evaluation.metrics import prompt_uptake


# ---------------------------------------------------------------------------
# Request schemas: bad input is a 4xx, never a 500 reachable deep in training
# ---------------------------------------------------------------------------
class TestRequestBounds:
    def test_train_request_defaults(self) -> None:
        req = TrainRequest()
        assert req.method == "lora"
        assert req.epochs == 3.0
        assert req.output_dir is None

    @pytest.mark.parametrize(
        "field,value",
        [
            ("method", "dragon"),
            ("epochs", -1.0),
            ("epochs", 0.0),
            ("learning_rate", -0.1),
            ("batch_size", 0),
            ("seq_len", 0),
            ("gradient_accumulation_steps", 0),
        ],
    )
    def test_train_request_rejects_bad_values(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError):
            TrainRequest(**{field: value})

    def test_dataset_request_rejects_bad_split_mode(self) -> None:
        with pytest.raises(ValidationError):
            DatasetPrepareRequest(split_mode="nonsense")
        req = DatasetPrepareRequest(split_mode="chronological")
        assert req.split_mode == "chronological"

    def test_dataset_request_rejects_out_of_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DatasetPrepareRequest(context_messages=99)
        with pytest.raises(ValidationError):
            DatasetPrepareRequest(test_fraction=0.9)
        with pytest.raises(ValidationError):
            DatasetPrepareRequest(max_context_chars=10)


# ---------------------------------------------------------------------------
# Held-out prompts: context-free examples are skipped and counted
# ---------------------------------------------------------------------------
def _example(rows: list[tuple[str, str]], convo: str = "c1"):
    from echotwin.dataset.builder import ChatTurn, ExampleMetadata, TrainingExample

    return TrainingExample(
        messages=[ChatTurn(role=r, content=t) for r, t in rows],
        metadata=ExampleMetadata(
            conversation_id=convo,
            language="en",
            example_type="natural_reply",
        ),
    )


class TestHeldoutPrompts:
    def test_context_free_examples_are_skipped_and_counted(self) -> None:
        with_context = _example([("user", "hi"), ("assistant", "hello there")])
        free = [_example([("assistant", "explicit opening line")]) for _ in range(3)]
        examples = [*free, with_context]
        assert count_context_free(examples) == 3
        prompts = heldout_prompts(examples, max_prompts=10)
        assert len(prompts) == 1
        assert prompts[0]["reference"] == "hello there"

    def test_count_is_zero_on_normal_examples(self) -> None:
        examples = [_example([("user", "hey"), ("assistant", "hey back")])]
        assert count_context_free(examples) == 0


# ---------------------------------------------------------------------------
# compare_runs: silently-empty alignment became an explicit error
# ---------------------------------------------------------------------------
class TestCompareRuns:
    def test_disjoint_runs_raise(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text(json.dumps({"index": 0, "generated": "x"}) + "\n", encoding="utf-8")
        b.write_text(json.dumps({"index": 1, "generated": "y"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="share no prompt indices"):
            compare_runs([str(a), str(b)])

    def test_overlapping_runs_align(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text(json.dumps({"index": 0, "generated": "x"}) + "\n", encoding="utf-8")
        b.write_text(
            json.dumps({"index": 0, "generated": "y"})
            + "\n"
            + json.dumps({"index": 9, "generated": "z"})
            + "\n",
            encoding="utf-8",
        )
        cmp_ = compare_runs([str(a), str(b)])
        assert cmp_["aligned_prompt_indices"] == [0]


class TestPromptUptake:
    def test_content_word_echo_measured(self) -> None:
        prompts = ["How was canteen paneer tikka?"]
        reps = ["canteen paneer tikka is my favorite"]
        r = prompt_uptake(prompts, reps)
        assert r["mean_uptake"] == 1.0
        assert r["zero_uptake_share"] == 0.0

    def test_generic_reply_scores_zero(self) -> None:
        r = prompt_uptake(
            ["How was canteen paneer tikka?"],
            ["I'm doing okay, thanks. How about you?"],
        )
        assert r["mean_uptake"] == 0.0
        assert r["zero_uptake_share"] == 1.0

    def test_stopwords_are_not_punished(self) -> None:
        # The prompt is all stopwords; uptake must not be -1 or nonsense, it
        # reports that the prompt carried no content words.
        r = prompt_uptake(["You did the thing"], ["I did it the"])
        assert r["mean_uptake"] is None or r["mean_uptake"] >= 0


# ---------------------------------------------------------------------------
# Jobs: stale rows from a crashed process become terminal, queued rows are
# cancellable from a fresh process
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


class TestStaleJobs:
    def test_fail_interrupted_terminalises_queued_and_running(self) -> None:
        from echotwin.workspace import get_store

        store = get_store()
        store.job_insert({"job_id": "j-queued", "kind": "k", "created_at": _now_iso(), "status": "queued"})
        store.job_insert({"job_id": "j-running", "kind": "k", "created_at": _now_iso(), "status": "running"})
        n = store.fail_interrupted()
        assert n == 2
        assert store.job_get("j-queued")["status"] == "failed"
        assert store.job_get("j-running")["status"] == "failed"

    def test_queued_job_of_dead_process_is_cancellable(self) -> None:
        """A queued row with no in-memory event must still be cancellable."""
        from echotwin.services import jobs
        from echotwin.workspace import get_store

        store = get_store()
        store.job_insert({"job_id": "j-dead", "kind": "k", "created_at": _now_iso(), "status": "queued"})
        assert jobs.cancel("j-dead") is True
        assert store.job_get("j-dead")["status"] == "cancelled"

    def test_running_job_no_event_rejected(self) -> None:
        """A 'running' row without an event belongs to another live process."""
        from echotwin.services import jobs
        from echotwin.workspace import get_store

        store = get_store()
        store.job_insert({"job_id": "j-other", "kind": "k", "created_at": _now_iso(), "status": "running"})
        assert jobs.cancel("j-other") is False


# ---------------------------------------------------------------------------
# Consent: revocation keeps the grant evidence; unknown scopes are rejected
# ---------------------------------------------------------------------------
class TestConsentEvidence:
    def test_revoke_preserves_grant_record(self) -> None:
        from echotwin.storage.db import get_consent
        from echotwin.voice.consent import grant_consent, revoke_consent

        grant_consent("whatsapp_import", {"method": "cli", "file_hash": "abc"})
        revoke_consent("whatsapp_import")
        row = get_consent("whatsapp_import")
        assert row["exists"] and not row["granted"]
        assert row["granted_at"] is not None, "revocation erased the grant timestamp"
        details = row["details"]
        assert details.get("method") == "cli", "grant details were lost on revoke"
        assert "revoked_at" in details

    def test_re_grant_updates(self) -> None:
        from echotwin.storage.db import get_consent
        from echotwin.voice.consent import grant_consent, revoke_consent

        grant_consent("voice_cloning", {"who": "self"})
        revoke_consent("voice_cloning")
        grant_consent("voice_cloning", {"who": "legal-guardian"})
        row = get_consent("voice_cloning")
        assert row["granted"] is True
        assert row["details"]["who"] == "legal-guardian"

    def test_unknown_scope_rejected(self) -> None:
        from echotwin.voice.consent import grant_consent, revoke_consent

        with pytest.raises(ValueError, match="Unknown consent scope"):
            grant_consent("typo_scope", {})
        with pytest.raises(ValueError, match="Unknown consent scope"):
            revoke_consent("typo_scope")


# ---------------------------------------------------------------------------
# Privacy export: the note must not claim the user's own words are absent
# ---------------------------------------------------------------------------
class TestPrivacyExportHonesty:
    def test_export_contains_own_words_note(self, tmp_path: Path) -> None:
        from echotwin.privacy_exporter import export_all

        out = tmp_path / "out"
        export_all(out, console=_QuietConsole())
        payload = json.loads(
            (out / "echotwin-state.json").read_text(encoding="utf-8")
        )
        note = payload.get("contents_note", "")
        assert "DOES contain your own words" in note
        assert "pointers" in payload and "workspace" in payload


class _QuietConsole:
    def print(self, *args: Any, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# doctor: a present-but-broken core dependency counts against health
# ---------------------------------------------------------------------------
class TestDoctorBrokenCore:
    def test_blocked_module_is_broken_not_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import echotwin.check_env as ce

        monkeypatch.setattr(
            ce, "_module_status", lambda name: "blocked (OSError)" if name == "numpy" else "installed"
        )
        report = ce.doctor_report()
        assert not report["healthy"]
        assert any("BROKEN" in line and "numpy" in line for line in report["lines"])

    def test_healthy_install_reports_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import echotwin.check_env as ce

        monkeypatch.setattr(ce, "_module_status", lambda name: "3.1")
        report = ce.doctor_report()
        assert report["healthy"]


# ---------------------------------------------------------------------------
# Voice: alternate backend availability requires the package AND cached weights
# ---------------------------------------------------------------------------
class TestChatterboxAvailability:
    def test_unavailable_without_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from echotwin.voice.alternative_backend import ChatterboxMultilingualBackend

        monkeypatch.setattr(
            ChatterboxMultilingualBackend, "_package_installed", lambda self: False
        )
        assert not ChatterboxMultilingualBackend().is_available()

    def test_unavailable_without_cached_weights(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from echotwin.voice.alternative_backend import ChatterboxMultilingualBackend

        monkeypatch.setattr(
            ChatterboxMultilingualBackend, "_package_installed", lambda self: True
        )
        monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
        backend = ChatterboxMultilingualBackend()
        assert not backend.is_available()

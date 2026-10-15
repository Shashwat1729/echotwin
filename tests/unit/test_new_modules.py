"""Unit tests: retrieval (incl. leakage exclusion), contamination, pairwise, generator."""

from __future__ import annotations

from echotwin.dataset.builder import TrainingExample
from echotwin.dataset.retrieval import (
    BM25Index,
    RetrievalConfig,
    StyleExemplar,
    build_index_from_examples,
    retrieve_exemplars,
)
from echotwin.evaluation.contamination import memorization_overlap
from echotwin.evaluation.pairwise import (
    HumanJudgment,
    PairwiseSample,
    build_pairs,
    judge_with_model,
    load_judgments,
    save_judgments,
    summarize,
)
from echotwin.models.generation import GeneratorConfig, MockGenerator, TransformersGenerator
from echotwin.models.registry import get_model_spec


def _ex(ex_id, context, response, convo):
    return TrainingExample.model_validate(
        {
            "messages": [
                {"role": "user", "content": context},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "conversation_id": convo,
                "language": "hi-Latn",
                "example_type": "natural_reply",
            },
        }
    )


class TestBM25:
    def test_add_and_len(self):
        idx = BM25Index()
        idx.add(StyleExemplar(doc_id="1", context="hello", response="hi", conversation_id="c1"))
        idx.finalize()
        assert len(idx) == 1

    def test_search_ranks_relevant(self):
        idx = BM25Index()
        idx.add(
            StyleExemplar(
                doc_id="1",
                context="canteen ka menu",
                response="paneer tikka hai kya",
                conversation_id="c1",
            )
        )
        idx.add(
            StyleExemplar(
                doc_id="2",
                context="weather today",
                response="baarish ho rahi hai",
                conversation_id="c2",
            )
        )
        idx.finalize()
        hits = idx.search("canteen menu kya hai", k=2)
        assert hits
        assert hits[0][0].doc_id == "1"

    def test_never_fabricates_when_no_match(self):
        idx = BM25Index()
        idx.add(StyleExemplar(doc_id="1", context="abc", response="def", conversation_id="c1"))
        idx.finalize()
        assert idx.search("zzz qqq", k=3) == []


class TestRetrievalLeakage:
    def test_excludes_current_conversation(self):
        examples = [
            _ex("e1", "canteen ka menu badal gaya", "serious? paneer tikka lao", "chat-a"),
            _ex("e2", "canteen ka menu badal gaya phir se", "paneer tikka time", "chat-b"),
        ]
        idx = build_index_from_examples(examples)
        prompt = [{"role": "user", "content": "canteen ka menu badal gaya"}]
        out = retrieve_exemplars(idx, prompt, conversation_id="chat-a", cfg=RetrievalConfig(k=3))
        assert out
        assert all(e["conversation_id"] != "chat-a" for e in out), (
            "retrieval must not leak the eval conversation"
        )

    def test_k_zero_returns_empty(self):
        idx = build_index_from_examples([_ex("e1", "hi", "hello", "c1")])
        assert (
            retrieve_exemplars(idx, [{"role": "user", "content": "hi"}], cfg=RetrievalConfig(k=0))
            == []
        )


class TestContamination:
    def test_exact_copy_detected(self):
        r = memorization_overlap(["haan yaar kal milte h", "kuch aur"], ["haan yaar kal milte h"])
        assert r.exact_match_rate == 0.5
        assert r.verdict in {"some_overlap", "high_overlap"}

    def test_clean_when_different(self):
        r = memorization_overlap(
            ["kal 6 bje library milte hain na"],
            ["paneer tikka kaisa tha bhai?", "aaj mausam badhiya hai"],
        )
        assert r.exact_match_rate == 0.0
        assert r.verdict == "clean"

    def test_no_train_data_reported(self):
        r = memorization_overlap(["abc def"], [])
        assert r.n_train_reference == 0
        assert r.verdict == "clean"

    def test_flags_without_full_text(self):
        r = memorization_overlap(["aa bb cc dd ee ff"], ["aa bb cc dd ee ff"])
        assert r.flagged_examples
        assert "diagnosis" in r.flagged_examples[0]


class TestPairwise:
    def test_summarize_counts(self):
        js = [
            HumanJudgment(sample_id="1", winner="a"),
            HumanJudgment(sample_id="2", winner="b"),
            HumanJudgment(sample_id="3", winner="a"),
            HumanJudgment(sample_id="4", winner="tie"),
        ]
        r = summarize(js)
        assert r.wins_a == 2 and r.wins_b == 1 and r.ties == 1
        assert r.win_rate_a == round(2 / 3, 4)

    def test_judge_with_model_deterministic(self):
        gen = MockGenerator(exemplar_texts=["A"], seed=1)
        sample = PairwiseSample(output_a="x", output_b="y", prompt_context=[])
        j = judge_with_model(gen, sample, GeneratorConfig(max_new_tokens=4))
        assert j.winner in {"a", "b", "tie"}
        assert "model-judge" in j.annotator

    def test_judgments_roundtrip(self, tmp_path):
        js = [HumanJudgment(sample_id="1", winner="a")]
        p = tmp_path / "j.jsonl"
        save_judgments(js, p)
        assert load_judgments(p)[0].winner == "a"

    def test_build_pairs_aligns(self):
        prompts = [
            {"messages": [], "reference": "r1", "language": "en"},
            {"messages": [], "reference": "r2", "language": "en"},
        ]
        outs = {"m1": ["a1", "a2"], "m2": ["b1"]}
        pairs = build_pairs(prompts, outs, "m1", "m2")
        assert len(pairs) == 1
        assert pairs[0].output_a == "a1"


class TestGeneratorThinkingMode:
    def test_thinking_stripped(self):
        from echotwin.models.generation import _TT_CLOSE, _TT_OPEN

        text = _TT_OPEN + "let me think about this" + _TT_CLOSE + "\nok bhai"
        stripped = TransformersGenerator._strip_thinking(text)
        assert _TT_OPEN not in stripped
        assert "ok bhai" in stripped

    def test_template_kwargs_for_switchable(self):
        from types import SimpleNamespace

        spec = get_model_spec("qwen3-4b")  # hybrid thinking model
        assert spec.thinking_mode == "switchable"
        g = TransformersGenerator(SimpleNamespace(tokenizer=None, spec=spec), spec)
        assert g._template_kwargs() == {"enable_thinking": False}

    def test_no_kwargs_for_non_thinking(self):
        from types import SimpleNamespace

        spec = get_model_spec("qwen3-4b-it-2507")
        assert spec.thinking_mode == "none"
        g = TransformersGenerator(SimpleNamespace(tokenizer=None, spec=spec), spec)
        assert g._template_kwargs() == {}

    def test_registry_default_is_verified_model(self):
        from echotwin.models.registry import DEFAULT_MODEL

        spec = get_model_spec(DEFAULT_MODEL)
        assert spec.hf_model_id == "Qwen/Qwen3-4B-Instruct-2507"
        assert spec.license == "Apache-2.0"
        assert spec.thinking_mode == "none"

    def test_sampling_resolution_uses_spec(self):
        from echotwin.models.registry import recommended_for_hindi

        spec = get_model_spec(recommended_for_hindi()[0])
        cfg = GeneratorConfig()
        resolved = cfg.resolve(spec)
        assert resolved["temperature"] == spec.recommended_sampling["temperature"]

"""Unit tests: training format/target-masking with a fake tokenizer, LoRA config."""

from __future__ import annotations

from echotwin.dataset.builder import TrainingExample
from echotwin.models.loader import resolve_target_modules
from echotwin.training.sft import (
    LoRAConfig,
    TrainingConfig,
    format_for_sft,
)


class FakeTokenizer:
    """Minimal tokenizer with chat-template support for masking tests."""

    def __init__(self):
        self.chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)

    def __call__(self, text, **kwargs):
        # 1 token per 4 chars, deterministic; honors truncation for realism
        max_length = kwargs.get("max_length") or kwargs.get("truncation", True)
        ids = list(range(100, 100 + max(1, len(text) // 4)))
        if isinstance(max_length, int):
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_target_masking():
    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "haan yaar"},
    ]
    enc = format_for_sft(tok, messages, 512)
    assert "input_ids" in enc and "labels" in enc
    # prompt tokens masked
    prompt_ids = tok(
        tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    )["input_ids"]
    assert enc["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    # target tokens NOT masked
    assert enc["labels"][len(prompt_ids) :] != [-100] * (len(enc["labels"]) - len(prompt_ids))


def test_truncation_respected():
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y" * 400}]
    enc = format_for_sft(tok, messages, 64)
    assert len(enc["input_ids"]) <= 64


class WordTokenizer:
    """Word-hash tokenizer where template drift breaks naive prefix masking."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "\n".join(parts)
        if not add_generation_prompt and len(messages) > 1:
            # simulate template whitespace/marker drift *inside* the prompt region,
            # so naive prefix-length masking would misplace the boundary
            text = text.replace("<|assistant|>", "driftmarker <|assistant|>", 1)
        return text

    def __call__(self, text, **kwargs):
        import hashlib

        ids = [int(hashlib.sha1(w.encode()).hexdigest()[:8], 16) % 5000 + 10 for w in text.split()]
        max_length = kwargs.get("max_length")
        if isinstance(max_length, int):
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _word_ids(text):
    import hashlib

    return [int(hashlib.sha1(w.encode()).hexdigest()[:8], 16) % 5000 + 10 for w in text.split()]


def test_masking_survives_template_drift():
    tok = WordTokenizer()
    messages = [
        {"role": "user", "content": "hello there friend"},
        {"role": "assistant", "content": "haan yaar milte hain"},
    ]
    enc = format_for_sft(tok, messages, 512)
    full_ids = enc["input_ids"]
    response_ids = _word_ids("haan yaar milte hain")
    # response span must be unmasked wherever it landed
    start = next(
        i
        for i in range(len(full_ids) - len(response_ids) + 1)
        if full_ids[i : i + len(response_ids)] == response_ids
    )
    assert enc["labels"][start : start + len(response_ids)] == response_ids
    assert all(lab == -100 for lab in enc["labels"][:start])
    assert enc["trainable_tokens"] == len(response_ids)


def test_unlocatable_target_raises():
    from echotwin.training.sft import TrainingError

    tok = WordTokenizer()
    with __import__("pytest").raises(TrainingError):
        format_for_sft(
            tok, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}], 512
        )


def test_train_all_assistant_turns():
    import hashlib

    class SpaceWordTokenizer(WordTokenizer):
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False, **kwargs
        ):
            parts = [f"<|{m['role']}|> {m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("<|assistant|>")
            return " ".join(parts)

    def wids(text):
        return [int(hashlib.sha1(w.encode()).hexdigest()[:8], 16) % 5000 + 10 for w in text.split()]

    tok = SpaceWordTokenizer()
    messages = [
        {"role": "user", "content": "pehla sawal"},
        {"role": "assistant", "content": "pehla jawab"},
        {"role": "user", "content": "doosra sawal"},
        {"role": "assistant", "content": "doosra jawab"},
    ]
    enc = format_for_sft(tok, messages, 512, train_on_last_only=False)
    full_ids = enc["input_ids"]
    first_ids, last_ids = wids("pehla jawab"), wids("doosra jawab")
    first_hit = next(
        i
        for i in range(len(full_ids) - len(first_ids) + 1)
        if full_ids[i : i + len(first_ids)] == first_ids
    )
    last_hit = next(
        i
        for i in range(len(full_ids) - len(last_ids) + 1)
        if full_ids[i : i + len(last_ids)] == last_ids
    )
    assert enc["labels"][first_hit : first_hit + len(first_ids)] == first_ids
    assert enc["labels"][last_hit : last_hit + len(last_ids)] == last_ids
    assert enc["labels"][0] == -100  # system/user prefix stays masked


def test_encode_split_skips_unmaskable():
    from echotwin.dataset.builder import ChatTurn, ExampleMetadata
    from echotwin.training.sft import _encode_split

    good = TrainingExample(
        messages=[ChatTurn(role="user", content="hi"), ChatTurn(role="assistant", content="hello")],
        metadata=ExampleMetadata(conversation_id="c", language="en", example_type="natural_reply"),
    )
    bad = TrainingExample(
        messages=[ChatTurn(role="user", content="hi"), ChatTurn(role="assistant", content="")],
        metadata=ExampleMetadata(conversation_id="c", language="en", example_type="natural_reply"),
    )
    items, skipped = _encode_split([good, bad], WordTokenizer(), 512)
    assert len(items) == 1 and skipped == 1


class _StubModule:
    """Minimal torch.nn.Module stand-in exposing only named_modules()."""

    def __init__(self):
        self._children: dict[str, _StubModule] = {}

    def add_child(self, name: str, module: _StubModule) -> None:
        self._children[name] = module

    def named_modules(self):
        yield "", self
        for name, child in self._children.items():
            yield name, child
            for sub_name, sub_module in child.named_modules():
                if sub_name:
                    yield f"{name}.{sub_name}", sub_module


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 16 and cfg.alpha == 32
        assert any("q_proj" in m for m in cfg.target_modules)

    def test_resolve_modules_matches_architecture(self):
        model = _StubModule()
        model.add_child("q_proj", _StubModule())
        model.add_child("o_proj", _StubModule())
        model.add_child("mlp", _StubModule())

        matched = resolve_target_modules(model, [".*q_proj", ".*o_proj"])
        assert "q_proj" in matched and "o_proj" in matched
        assert "mlp" not in matched

    def test_resolve_no_match_empty(self):
        model = _StubModule()

        assert resolve_target_modules(model, [".*q_proj"]) == []


class TestTrainingConfig:
    def test_qlora_flagged(self):
        cfg = TrainingConfig(output_dir="x", method="qlora")
        assert cfg.method == "qlora"

    def test_config_serializable(self):
        cfg = TrainingConfig(output_dir="x", method="lora")
        blob = cfg.model_dump_json()
        assert '"lr_scheduler"' in blob


def test_context_free_example_does_not_render_empty_prompt():
    """Conversation openers have no prompt turns; strict templates reject [] (seen on Qwen)."""

    class Strict(WordTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if not messages:
                raise ValueError("Cannot apply chat template to an empty conversation.")
            return super().apply_chat_template(messages, **kwargs)

    enc = format_for_sft(Strict(), [{"role": "assistant", "content": "haan yaar milte hain"}], 512)
    assert enc["trainable_tokens"] == len(_word_ids("haan yaar milte hain"))


def test_training_args_compat_renames_warmup_ratio(monkeypatch):
    import sys
    import types

    from echotwin.training.sft import _training_args_compat

    class V5Args:
        def __init__(self, **kw):
            if "warmup_ratio" in kw:
                raise TypeError("__init__() got an unexpected keyword argument 'warmup_ratio'")
            self.kw = kw

    fake = types.SimpleNamespace(TrainingArguments=V5Args)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    out = _training_args_compat(warmup_ratio=0.05, learning_rate=1e-4)
    assert out.kw == {"warmup_steps": 0.05, "learning_rate": 1e-4}

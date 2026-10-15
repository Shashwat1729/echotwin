"""Unit tests: CPU n-gram style model (train/generate/save/determinism)."""

from __future__ import annotations

import pytest

from echotwin.models.generation import GeneratorConfig
from echotwin.models.ngram import NGramConfig, NGramGenerator, NGramModel, tokenize

TEXTS = [
    "haan bro kal milte hain 6 bje",
    "haan bro kal milte hain library mein",
    "theek hai done, kal baat karte hain",
    "morning! ummeed hai aaj ka din achha jayega",
]


class TestNGram:
    def test_tokenize_preserves_case_and_emoji(self):
        toks = tokenize("Good Morning!! ☀️")
        assert "Good" in toks and "Morning" in toks
        assert "!" in toks

    def test_train_counts(self):
        m = NGramModel(order=3).train_texts(TEXTS)
        assert m.n_targets == 4
        assert len(m.vocab) > 5
        assert m.lengths and all(v > 0 for v in m.lengths)

    def test_empty_train(self):
        m = NGramModel(order=2).train_texts([])
        assert m.n_targets == 0
        assert m.generate("hello") == ""

    def test_generate_deterministic(self):
        m = NGramModel(order=3).train_texts(TEXTS)
        a = m.generate("kal milte", seed=1)
        b = m.generate("kal milte", seed=1)
        assert a == b and a.strip()

    def test_generate_uses_style_words(self):
        m = NGramModel(order=2).train_texts(TEXTS)
        outs = {m.generate("kal", seed=s) for s in range(10)}
        joined = " ".join(outs).lower()
        assert any(w in joined for w in ("haan", "kal", "theek", "milte"))

    def test_save_load_roundtrip(self, tmp_path):
        m = NGramModel(order=3).train_texts(TEXTS)
        p = m.save(tmp_path / "ngram.json")
        loaded = NGramModel.load(p)
        assert loaded.n_targets == m.n_targets
        assert loaded.generate("kal", seed=3) == m.generate("kal", seed=3)

    def test_order_validation(self):
        with pytest.raises(Exception):
            NGramModel(order=9)

    def test_generator_interface(self):
        m = NGramModel(order=2).train_texts(TEXTS)
        gen = NGramGenerator(m, seed=2)
        out = gen.generate([{"role": "user", "content": "kal milte ho?"}], GeneratorConfig())
        assert isinstance(out, str) and out.strip()

    def test_config_defaults(self):
        cfg = NGramConfig()
        assert cfg.order == 3 and cfg.top_k > 0

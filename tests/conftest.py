"""
Shared fixtures.

`fake_manager` overrides the real ModelManager dependency so test_api.py
can exercise the full request/response contract (routing, validation,
error handling) without downloading real checkpoints from the Hub --
that's what test_ml.py is for, using a tiny locally-instantiated GPT-2
config instead of the real fine-tuned weights.
"""
import pytest
from fastapi.testclient import TestClient
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from app.main import app
from app.api.routes import get_model_manager


class _FakeModelManager:
    """Drop-in replacement for ModelManager: same interface, tiny untrained
    model, no Hub network calls."""

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.vocab_size < 1000:
            # from_pretrained() degrades to an empty tokenizer instead of
            # raising when huggingface.co is unreachable -- fail loudly here
            # rather than silently building a broken 0-vocab model.
            raise RuntimeError(
                "GPT2Tokenizer loaded with a suspiciously small vocab "
                f"({self.tokenizer.vocab_size}) -- this usually means "
                "huggingface.co was unreachable. Check network access."
            )
        config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=self.tokenizer.vocab_size)
        self._model = GPT2LMHeadModel(config)
        self._model.eval()

    def get(self, topic, size):
        return self._model

    def resident_keys(self):
        return ("fake_model",)


@pytest.fixture(scope="session")
def fake_manager():
    return _FakeModelManager()


@pytest.fixture(scope="session")
def client(fake_manager):
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _patch_evaluator(monkeypatch):
    """API contract tests shouldn't need to download MiniLM or the real
    reference_texts.json from the Hub -- only test_ml.py exercises the
    real scoring path."""
    from app.services import evaluator

    fake_refs = {"World": ["ref"], "Sports": ["ref"], "Business": ["ref"], "Sci/Tech": ["ref"]}
    monkeypatch.setattr(evaluator, "get_references", lambda: fake_refs)
    monkeypatch.setattr(evaluator, "style_similarity", lambda refs, gens: 0.5)


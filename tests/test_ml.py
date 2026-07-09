"""
ML-level sanity checks for generator.py and evaluator.py, run against a
tiny locally-instantiated (untrained) GPT-2 config rather than the real
fine-tuned checkpoints -- these test *shapes and types*, not quality.

Note: these still need network access to download the "gpt2" tokenizer
vocab/merges files and the MiniLM embedder from the Hub the first time
they run (small, but not zero -- a few MB total). That's expected in a
normal CI runner with internet access; it will fail in a fully offline
sandbox.
"""
import pytest
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from app.services import evaluator, generator


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.vocab_size < 1000:
        pytest.skip(
            "GPT2Tokenizer loaded with a suspiciously small vocab "
            f"({tokenizer.vocab_size}) -- huggingface.co was likely "
            "unreachable from this environment, not a code bug."
        )
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=tokenizer.vocab_size)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, tokenizer


def test_generate_unconditional_returns_nonempty_string(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    text = generator.generate(model, tokenizer, prompt="")
    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_with_prompt_returns_string(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    text = generator.generate(model, tokenizer, prompt="Hello there")
    assert isinstance(text, str)


@pytest.mark.integration
def test_style_similarity_in_valid_cosine_range():
    """Marked integration: downloads the real MiniLM embedder."""
    score = evaluator.style_similarity(
        references=["a reference headline about sports"],
        generations=["a generated headline about sports"],
    )
    assert -1.0 <= score <= 1.0

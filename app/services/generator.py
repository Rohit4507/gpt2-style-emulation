"""Causal LM nucleus-sampling generation, matching the paper's Table I settings."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from app.core.config import settings


def generate(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str = "") -> str:
    """Unconditional decode if prompt is empty, else continue from prompt
    (used both for plain generation and for chained drift hops)."""
    if not prompt:
        input_ids = torch.full((1, 1), tokenizer.eos_token_id, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=True,
            top_k=settings.top_k,
            top_p=settings.top_p,
            temperature=settings.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

"""
content_gen.py – Flan-T5 product description generator.
Supports style prefixes: seo / short / luxury.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..data import build_generation_prompt

STYLE_PREFIXES: dict[str, str] = {
    "seo":     "Write an SEO-optimized product description: ",
    "short":   "Write a concise one-sentence product description: ",
    "luxury":  "Write a luxury, aspirational product description: ",
    "default": "Write a product description: ",
}


class ContentGenerator:
    """
    Seq2Seq product description generator backed by Flan-T5.

    Parameters
    ----------
    model_name_or_dir : str
        Hugging Face model ID or path to a fine-tuned checkpoint directory.
    device : str, optional
        'cuda' or 'cpu'. Auto-detected if not provided.
    """

    def __init__(
        self,
        model_name_or_dir: str = "google/flan-t5-small",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name_or_dir)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name_or_dir)
            .eval()
            .to(self.device)
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> str:
        """Generate text from *prompt* using beam search."""
        ids = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tok.decode(out[0], skip_special_tokens=True)

    def describe_product(
        self, product: dict, style: str = "default"
    ) -> str:
        """
        Generate a product description with an optional style prefix.

        Parameters
        ----------
        product : dict
            Must contain keys: name, category, brand, color, attributes.
        style : str
            One of 'seo', 'short', 'luxury', 'default'.
        """
        prefix = STYLE_PREFIXES.get(style, STYLE_PREFIXES["default"])
        base_prompt = build_generation_prompt(product)
        # Replace the generic prefix with the style-specific one
        prompt = prefix + base_prompt.split(": ", 1)[-1]
        return self.generate(prompt)

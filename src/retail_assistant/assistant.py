"""
assistant.py – RAG-based shopping assistant with conversation memory,
metadata filters parsed from natural language, and a grounded prompt
that refuses to invent products not in the retrieved context.
"""
from __future__ import annotations

import re
from collections import deque

from .models.embedder import ProductIndex
from .models.content_gen import ContentGenerator


# ── filter extraction ─────────────────────────────────────────────────────────
_RANGE_RE = re.compile(
    r"(?:between\s+)?\$?(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_MAX_PRICE_RE = re.compile(
    r"(?:under|below|less\s+than|up\s+to|max|maximum)\s*\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_MIN_PRICE_RE = re.compile(
    r"(?:above|over|more\s+than|at\s+least|min|minimum)\s*\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|green|grey|gray|navy|brown|beige|pink|yellow|purple|"
    r"orange|gold|silver|multicolour|multicolor)\b",
    re.IGNORECASE,
)
_BRAND_RE = re.compile(
    r"\b(?:by|from|brand)?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b"
)

KNOWN_CATEGORIES = {
    "topwear", "bottomwear", "shoes", "sandals", "watches", "bags",
    "eyewear", "sunglasses", "belts", "socks", "jewellery", "headwear",
}


def parse_filters(query: str) -> dict:
    """
    Extract structured filters from a free-text query.

    Returns a dict with keys: min_price, max_price, color, category, brand.
    All values default to None when not found.
    """
    filters: dict = {
        "min_price": None,
        "max_price": None,
        "color": None,
        "category": None,
        "brand": None,
    }

    # Price
    range_match = _RANGE_RE.search(query)
    if range_match:
        filters["min_price"] = float(range_match.group(1))
        filters["max_price"] = float(range_match.group(2))
    else:
        max_m = _MAX_PRICE_RE.search(query)
        if max_m:
            filters["max_price"] = float(max_m.group(1))
        min_m = _MIN_PRICE_RE.search(query)
        if min_m:
            filters["min_price"] = float(min_m.group(1))
        if filters["min_price"] is None and filters["max_price"] is None:
            price_match = re.search(r"\$(\d+(?:\.\d+)?)", query)
            if price_match:
                if any(w in query.lower() for w in ("under", "below", "less", "max")):
                    filters["max_price"] = float(price_match.group(1))
                else:
                    filters["min_price"] = float(price_match.group(1))

    # Color
    color_match = _COLOR_RE.search(query)
    if color_match:
        filters["color"] = color_match.group(1)

    # Category (exact keyword match)
    q_lower = query.lower()
    for cat in KNOWN_CATEGORIES:
        if cat in q_lower:
            filters["category"] = "eyewear" if cat == "sunglasses" else cat
            break

    return filters


# ── assistant ─────────────────────────────────────────────────────────────────
class ShoppingAssistant:
    """
    Retrieval-augmented shopping assistant.

    Features
    --------
    - Conversation memory (last ``memory_turns`` Q/A pairs)
    - Automatic filter extraction from natural language (price, color, category)
    - Grounded prompt that instructs the model not to invent products
    - ``similar_items()`` helper for product-to-product recommendations
    """

    def __init__(
        self,
        index: ProductIndex,
        generator: ContentGenerator | None = None,
        top_k: int = 5,
        memory_turns: int = 5,
        refuse_hallucination: bool = True,
    ):
        self.index = index
        self.gen = generator
        self.top_k = top_k
        self.refuse_hallucination = refuse_hallucination
        self.history: deque[tuple[str, str]] = deque(maxlen=memory_turns)

    # ── public API ─────────────────────────────────────────────────────────────
    def chat(self, message: str) -> dict:
        """
        Process one user message and return an answer plus retrieved products.

        Parameters
        ----------
        message : str
            User's natural-language message.

        Returns
        -------
        dict with keys: answer (str), products (list[dict]), filters (dict)
        """
        filters = parse_filters(message)
        hits = self.index.search_with_filter(
            message,
            k=self.top_k,
            category=filters.get("category"),
            color=filters.get("color"),
            min_price=filters.get("min_price"),
            max_price=filters.get("max_price"),
        )

        # Fall back to unfiltered search if filter left no results
        if len(hits) == 0:
            hits = self.index.search(message, self.top_k)

        context_lines = [
            f"- {r['name']} | ${r['price']:.2f} | {r['category']} | {r['color']} | {r['attributes']}"
            for _, r in hits.iterrows()
        ]
        context = "\n".join(context_lines)

        if self.gen:
            history_str = "\n".join(
                f"Customer: {q}\nAssistant: {a}" for q, a in self.history
            )
            grounding_note = (
                "\nIMPORTANT: Only recommend products listed above. "
                "Do not invent or mention any product not in the list.\n"
                if self.refuse_hallucination
                else ""
            )
            prompt = (
                f"You are a helpful retail shopping assistant.\n"
                f"Available products:\n{context}\n"
                f"{grounding_note}"
                f"{'Previous conversation:' + chr(10) + history_str + chr(10) if history_str else ''}"
                f"Customer: {message}\nAssistant:"
            )
            answer = self.gen.generate(prompt, max_new_tokens=150)
        else:
            if len(hits) == 0:
                answer = "I could not find any matching products in our catalog."
            else:
                top = hits.iloc[0]
                answer = (
                    f"I'd suggest the **{top['name']}** (${top['price']:.2f}): "
                    f"{top['attributes']}."
                )

        self.history.append((message, answer))
        return {
            "answer": answer,
            "products": hits.to_dict("records"),
            "filters": filters,
        }

    def similar_items(self, product_name: str, k: int = 5):
        """Find products similar to *product_name* by embedding its text."""
        return self.index.search(product_name, k)

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

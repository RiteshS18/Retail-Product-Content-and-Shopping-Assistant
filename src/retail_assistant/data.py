"""
data.py – data loading utilities for the retail assistant.
"""
from __future__ import annotations

import pandas as pd
from .config import load_config, abs_path


def load_products(path: str | None = None) -> pd.DataFrame:
    """
    Load the product catalog CSV.

    Priority:
    1. The explicit *path* argument.
    2. ``paths.processed_csv`` in config (real data).
    3. ``paths.products_csv`` in config (legacy sample).
    """
    cfg = load_config()
    if path is not None:
        p = abs_path(path)
    else:
        # prefer processed data if it exists
        processed = abs_path(cfg["paths"].get("processed_csv", ""))
        if processed.exists():
            p = processed
        else:
            p = abs_path(cfg["paths"]["products_csv"])
    df = pd.read_csv(p).fillna("")
    return df


def build_generation_prompt(row) -> str:
    """
    Build the input prompt for the content generator.

    Parameters
    ----------
    row : dict-like (pd.Series or dict)
        Must have keys: name, category, brand, color, attributes.
    """
    return (
        f"Write a product description. "
        f"Name: {row['name']}. "
        f"Category: {row['category']}. "
        f"Brand: {row['brand']}. "
        f"Color: {row['color']}. "
        f"Features: {row['attributes']}."
    )


def product_to_text(row) -> str:
    """
    Produce a single text string used for embedding / retrieval.

    Parameters
    ----------
    row : dict-like
        Must have keys: name, category, brand, color, attributes, price.
    """
    return (
        f"{row['name']} | {row['category']} | {row['brand']} | "
        f"{row['color']} | {row['attributes']} | ${row['price']}"
    )

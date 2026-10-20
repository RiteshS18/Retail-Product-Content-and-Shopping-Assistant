"""
build_index.py
==============
Encode the product catalog and build the FAISS search index.

Usage
-----
    python -m train.build_index [--csv data/processed/products.csv]

Outputs
-------
models_out/index/emb.npy       – normalized embeddings
models_out/index/catalog.csv   – product catalog copy
models_out/index/faiss.index   – FAISS FlatIP index (if faiss installed)
reports/search_results.json    – Recall@k / MRR evaluation
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from retail_assistant.config import load_config, abs_path
from retail_assistant.data import load_products, product_to_text
from retail_assistant.models.embedder import ProductIndex


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate_search(index: ProductIndex, df: pd.DataFrame, seed: int = 42) -> dict:
    """
    Evaluate Recall@1/5/10 and MRR using category-match as relevance signal.

    For each test query (product name), a hit is relevant if the retrieved
    product belongs to the same category as the query product.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Sample 200 test queries
    sample = df.sample(min(200, len(df)), random_state=seed)
    K_VALUES = [1, 5, 10]
    recall: dict[int, list[float]] = {k: [] for k in K_VALUES}
    reciprocal_ranks: list[float] = []

    for _, row in sample.iterrows():
        query = product_to_text(row)
        hits = index.search(query, k=max(K_VALUES))
        # Exclude exact match (same product_id if present)
        pid = row.get("product_id", None)
        hits_filtered = (
            hits[hits["product_id"] != pid]
            if pid is not None and "product_id" in hits.columns
            else hits
        )
        retrieved_cats = hits_filtered["category"].tolist()
        true_cat = str(row["category"])

        for k in K_VALUES:
            top_k_cats = retrieved_cats[:k]
            recall[k].append(float(true_cat in top_k_cats))

        # MRR: rank of first relevant result
        rr = 0.0
        for rank, cat in enumerate(retrieved_cats, start=1):
            if cat == true_cat:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    results = {
        "num_queries": len(sample),
        "MRR": round(float(np.mean(reciprocal_ranks)), 4),
    }
    for k in K_VALUES:
        results[f"Recall@{k}"] = round(float(np.mean(recall[k])), 4)
    return results


# ── main ───────────────────────────────────────────────────────────────────────
def main(csv_path: str | None = None) -> None:
    cfg = load_config()
    seed = cfg["seed"]

    df = load_products(csv_path)
    print(f"Loaded {len(df):,} products.")

    use_faiss = cfg.get("embedder", {}).get("use_faiss", True)
    idx = ProductIndex(cfg["embedder"]["model_name"], use_faiss=use_faiss)
    idx.build(df)

    out_dir = abs_path(cfg["paths"]["index_dir"])
    idx.save(out_dir)
    print(f"Index saved -> {out_dir}")

    # ── search evaluation ──────────────────────────────────────────────────────
    print("Evaluating search (Recall@k, MRR) ...")
    metrics = evaluate_search(idx, df, seed=seed)
    print(json.dumps(metrics, indent=2))

    reports_dir = abs_path(cfg["paths"].get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "search_results.json").write_text(json.dumps(metrics, indent=2))
    print(f"Search metrics -> {reports_dir / 'search_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build product search index")
    parser.add_argument("--csv", default=None, help="Path to products CSV (overrides config)")
    args = parser.parse_args()
    main(args.csv)

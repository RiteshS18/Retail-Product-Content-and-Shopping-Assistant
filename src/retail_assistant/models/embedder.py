"""
embedder.py – dense product retrieval with FAISS and numpy fallback.
Supports text-only (sentence-transformers) search.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from ..data import product_to_text


class ProductIndex:
    """
    Dense retrieval over the product catalog.

    Uses FAISS flat index when available (cfg embedder.use_faiss=true),
    falls back to numpy dot-product cosine similarity otherwise.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_faiss: bool = True,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.use_faiss = use_faiss
        self.emb: np.ndarray | None = None
        self.df = None
        self._faiss_index = None

    # ── build ──────────────────────────────────────────────────────────────────
    def build(self, df) -> "ProductIndex":
        """Encode all products and build the retrieval index."""
        import pandas as pd  # noqa: PLC0415

        self.df = df.reset_index(drop=True)
        texts = [product_to_text(r) for _, r in self.df.iterrows()]
        self.emb = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
        ).astype(np.float32)

        if self.use_faiss:
            self._build_faiss()
        return self

    def _build_faiss(self) -> None:
        """Build a FAISS FlatIP index (cosine on unit vectors = inner product)."""
        try:
            import faiss  # noqa: PLC0415

            d = self.emb.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(self.emb)
            self._faiss_index = index
        except ImportError:
            print("[WARN] faiss not installed – falling back to numpy search.")
            self._faiss_index = None

    # ── search ─────────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5):
        """
        Search the index for the top-k most relevant products.

        Parameters
        ----------
        query : str
            Natural-language query.
        k : int
            Number of results to return.

        Returns
        -------
        pd.DataFrame
            Subset of self.df with an added 'score' column.
        """
        import pandas as pd  # noqa: PLC0415

        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q[np.newaxis], k)
            top_idx = indices[0]
            top_scores = scores[0]
        else:
            scores_all = self.emb @ q
            top_idx = np.argsort(-scores_all)[:k]
            top_scores = scores_all[top_idx]

        res = self.df.iloc[top_idx].copy()
        res["score"] = top_scores
        return res

    def search_with_filter(
        self,
        query: str,
        k: int = 5,
        category: str | None = None,
        color: str | None = None,
        brand: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ):
        """Search with optional metadata filters applied after retrieval."""
        # Retrieve extra candidates for post-filter
        candidates = self.search(query, k=min(len(self.df), k * 10))

        if category:
            candidates = candidates[
                candidates["category"].str.lower() == category.lower()
            ]
        if color:
            candidates = candidates[
                candidates["color"].str.lower().str.contains(color.lower(), na=False)
            ]
        if brand:
            candidates = candidates[
                candidates["brand"].str.lower().str.contains(brand.lower(), na=False)
            ]
        if min_price is not None:
            candidates = candidates[candidates["price"] >= min_price]
        if max_price is not None:
            candidates = candidates[candidates["price"] <= max_price]

        return candidates.head(k)

    # ── persist ────────────────────────────────────────────────────────────────
    def save(self, directory: str | Path) -> None:
        """Persist embeddings and catalog to disk."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb.npy", self.emb)
        self.df.to_csv(d / "catalog.csv", index=False)
        # Save FAISS index if available
        if self._faiss_index is not None:
            try:
                import faiss  # noqa: PLC0415

                faiss.write_index(self._faiss_index, str(d / "faiss.index"))
            except ImportError:
                pass

    @classmethod
    def load(
        cls,
        directory: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_faiss: bool = True,
    ) -> "ProductIndex":
        """Load a previously saved index from disk."""
        import pandas as pd  # noqa: PLC0415

        d = Path(directory)
        obj = cls(model_name, use_faiss=use_faiss)
        obj.emb = np.load(d / "emb.npy").astype(np.float32)
        obj.df = pd.read_csv(d / "catalog.csv").fillna("")

        faiss_path = d / "faiss.index"
        if use_faiss and faiss_path.exists():
            try:
                import faiss  # noqa: PLC0415

                obj._faiss_index = faiss.read_index(str(faiss_path))
            except ImportError:
                obj._build_faiss()
        else:
            obj._build_faiss()

        return obj

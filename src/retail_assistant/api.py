"""
api.py – FastAPI backend for the Retail Shopping Assistant.

Endpoints
---------
GET  /health          liveness check
POST /search          text search with optional k parameter
POST /chat            shopping assistant chat
POST /generate        generate product description
POST /classify        classify a product image -> top-3 categories
POST /similar         find similar products by name
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from .config import load_config, abs_path
from .models.embedder import ProductIndex
from .models.content_gen import ContentGenerator
from .assistant import ShoppingAssistant

logger = logging.getLogger("retail_assistant.api")

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Product Content & Shopping Assistant",
    description="Four-module retail AI: classifier · content-gen · semantic-search · RAG chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config()


# ── cached model loading ──────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_index() -> ProductIndex:
    """Load the search index once and cache it."""
    d = abs_path(cfg["paths"]["index_dir"])
    if not (d / "emb.npy").exists():
        raise HTTPException(
            status_code=503,
            detail="Search index not found. Run: python -m train.build_index",
        )
    use_faiss = cfg.get("embedder", {}).get("use_faiss", True)
    return ProductIndex.load(d, cfg["embedder"]["model_name"], use_faiss=use_faiss)


@lru_cache(maxsize=1)
def get_generator() -> ContentGenerator:
    """Load the content generator once and cache it."""
    d = abs_path(cfg["paths"]["generator_dir"])
    model_path = str(d) if d.exists() else cfg["generator"]["base_model"]
    return ContentGenerator(model_path)


@lru_cache(maxsize=1)
def get_assistant() -> ShoppingAssistant:
    """Build a ShoppingAssistant with cached index and generator."""
    memory = cfg.get("A", {}).get("memory_turns", 5)
    refuse = cfg.get("A", {}).get("refuse_hallucination", True)
    try:
        gen = get_generator()
    except Exception:
        gen = None
    return ShoppingAssistant(
        get_index(), gen, cfg["embedder"]["top_k"],
        memory_turns=memory, refuse_hallucination=refuse,
    )


# ── request / response schemas ────────────────────────────────────────────────
class Query(BaseModel):
    query: str = Field(..., min_length=1, max_length=512, json_schema_extra={"example": "black running shoes under $100"})
    k: int = Field(5, ge=1, le=50)


class ProductIn(BaseModel):
    name: str = Field(..., min_length=1, json_schema_extra={"example": "Nova Running Shoes - Black"})
    category: str = Field("", json_schema_extra={"example": "shoes"})
    brand: str = Field("", json_schema_extra={"example": "Nova"})
    color: str = Field("", json_schema_extra={"example": "black"})
    attributes: str = Field("", json_schema_extra={"example": "lightweight; breathable mesh; cushioned sole"})
    style: str = Field("default", json_schema_extra={"example": "seo"})


class SimilarQuery(BaseModel):
    product_name: str = Field(..., json_schema_extra={"example": "Nova Running Shoes - Black"})
    k: int = Field(5, ge=1, le=20)


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    """Liveness check."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/search", tags=["search"])
def search(q: Query):
    """
    Semantic product search.

    Returns the *k* most relevant products for the given *query*.
    """
    try:
        results = get_index().search(q.query, q.k)
        return results.to_dict("records")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", tags=["assistant"])
def chat(q: Query):
    """
    Shopping assistant chat endpoint.

    Returns an AI-generated answer grounded in the product catalog, plus
    the retrieved products and any parsed filters.
    """
    try:
        return get_assistant().chat(q.query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate", tags=["content"])
def generate(p: ProductIn):
    """Generate a product description from structured attributes."""
    try:
        description = get_generator().describe_product(p.model_dump(), style=p.style)
        return {"description": description}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Generate error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/classify", tags=["classifier"])
async def classify(file: UploadFile = File(...)):
    """
    Classify a product image.

    Returns top-3 predicted categories with confidence scores.
    """
    from .models.classifier import CategoryClassifier  # noqa: PLC0415

    ckpt = abs_path(cfg["paths"]["classifier_ckpt"])
    if not ckpt.exists():
        raise HTTPException(
            status_code=503,
            detail="Classifier checkpoint not found. Run: python -m train.train_classifier",
        )

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Invalid image file.") from exc

    try:
        clf = CategoryClassifier(ckpt)
        predictions = clf.predict(img)
        return {
            "predictions": [
                {"category": cat, "confidence": round(conf, 4)}
                for cat, conf in predictions
            ]
        }
    except Exception as exc:
        logger.exception("Classify error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/similar", tags=["search"])
def similar(q: SimilarQuery):
    """Find products similar to a given product name."""
    try:
        results = get_index().search(q.product_name, q.k)
        return results.to_dict("records")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Similar error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/chat/history", tags=["assistant"])
def reset_chat():
    """Clear the shopping assistant's conversation history."""
    try:
        get_assistant().reset()
        return {"message": "History cleared."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

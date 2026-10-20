"""
test_suite.py – comprehensive pytest tests for all four modules + API.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def sample_df():
    from retail_assistant.data import load_products  # noqa: PLC0415
    return load_products()


@pytest.fixture(scope="session")
def tiny_df():
    """A minimal in-memory dataframe for fast tests."""
    return pd.DataFrame([
        {
            "product_id": 1, "name": "Nova Running Shoes - Black", "category": "shoes",
            "brand": "Nova", "color": "black", "price": 79.99,
            "attributes": "lightweight; breathable", "description": "Great shoes.",
            "image_path": "shoes/1.jpg",
        },
        {
            "product_id": 2, "name": "Aero Topwear - White", "category": "topwear",
            "brand": "Aero", "color": "white", "price": 39.99,
            "attributes": "cotton blend; slim fit", "description": "Smart top.",
            "image_path": "topwear/2.jpg",
        },
        {
            "product_id": 3, "name": "Zenith Watch - Silver", "category": "watches",
            "brand": "Zenith", "color": "silver", "price": 199.00,
            "attributes": "quartz; water resistant", "description": "Classic watch.",
            "image_path": "watches/3.jpg",
        },
    ])


@pytest.fixture(scope="session")
def product_index(tiny_df):
    from retail_assistant.models.embedder import ProductIndex  # noqa: PLC0415
    idx = ProductIndex(use_faiss=False)  # numpy – no faiss required
    idx.build(tiny_df)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Data module tests
# ─────────────────────────────────────────────────────────────────────────────
class TestData:
    def test_load_products(self, sample_df):
        assert len(sample_df) > 0

    def test_required_columns(self, sample_df):
        required = {"name", "category", "brand", "color", "price", "attributes", "description"}
        assert required.issubset(set(sample_df.columns))

    def test_build_generation_prompt(self, sample_df):
        from retail_assistant.data import build_generation_prompt  # noqa: PLC0415
        prompt = build_generation_prompt(sample_df.iloc[0])
        assert "Write a product description" in prompt
        assert sample_df.iloc[0]["name"] in prompt

    def test_product_to_text(self, sample_df):
        from retail_assistant.data import product_to_text  # noqa: PLC0415
        text = product_to_text(sample_df.iloc[0])
        assert sample_df.iloc[0]["name"] in text
        assert "$" in text

    def test_no_null_in_required_cols(self, sample_df):
        for col in ("name", "category"):
            assert sample_df[col].notna().all(), f"Nulls found in column: {col}"


# ─────────────────────────────────────────────────────────────────────────────
# Search / Embedder tests
# ─────────────────────────────────────────────────────────────────────────────
class TestEmbedder:
    def test_build_index(self, product_index, tiny_df):
        assert product_index.emb is not None
        assert product_index.emb.shape[0] == len(tiny_df)

    def test_search_returns_df(self, product_index):
        results = product_index.search("running shoes", k=2)
        assert len(results) <= 2
        assert "name" in results.columns
        assert "score" in results.columns

    def test_search_top_result_relevant(self, product_index):
        results = product_index.search("running shoes black", k=3)
        assert results.iloc[0]["category"] == "shoes"

    def test_save_load(self, product_index, tmp_path):
        from retail_assistant.models.embedder import ProductIndex  # noqa: PLC0415
        product_index.save(tmp_path)
        loaded = ProductIndex.load(tmp_path, use_faiss=False)
        assert loaded.emb.shape == product_index.emb.shape
        assert len(loaded.df) == len(product_index.df)

    def test_filter_search(self, product_index):
        results = product_index.search_with_filter("shoes", k=3, category="shoes")
        assert all(r == "shoes" for r in results["category"].tolist())

    def test_search_with_price_filter(self, product_index):
        results = product_index.search_with_filter("any product", k=5, max_price=50.0)
        assert all(p <= 50.0 for p in results["price"].tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Content generator tests
# ─────────────────────────────────────────────────────────────────────────────
class TestContentGenerator:
    @pytest.fixture(scope="class")
    def generator(self):
        from retail_assistant.models.content_gen import ContentGenerator  # noqa: PLC0415
        return ContentGenerator()  # zero-shot Flan-T5-small

    def test_generate_returns_string(self, generator):
        out = generator.generate("Write a short product description: running shoes.")
        assert isinstance(out, str)
        assert len(out) > 5

    def test_describe_product(self, generator):
        product = {
            "name": "Nova Running Shoes", "category": "shoes",
            "brand": "Nova", "color": "black", "attributes": "lightweight; breathable",
        }
        desc = generator.describe_product(product)
        assert isinstance(desc, str) and len(desc) > 5

    def test_style_prefixes(self, generator):
        from retail_assistant.models.content_gen import STYLE_PREFIXES  # noqa: PLC0415
        product = {
            "name": "Test Bag", "category": "bags",
            "brand": "TestBrand", "color": "red", "attributes": "durable; spacious",
        }
        for style in STYLE_PREFIXES:
            desc = generator.describe_product(product, style=style)
            assert isinstance(desc, str)


# ─────────────────────────────────────────────────────────────────────────────
# Assistant tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAssistant:
    @pytest.fixture(scope="class")
    def assistant(self, product_index):
        from retail_assistant.assistant import ShoppingAssistant  # noqa: PLC0415
        return ShoppingAssistant(product_index, generator=None, top_k=3, memory_turns=3)

    def test_chat_returns_dict(self, assistant):
        resp = assistant.chat("I need running shoes")
        assert "answer" in resp
        assert "products" in resp
        assert isinstance(resp["products"], list)

    def test_memory_accumulates(self, assistant):
        assistant.reset()
        assistant.chat("Show me shoes")
        assistant.chat("What about bags?")
        assert len(assistant.history) == 2

    def test_memory_bounded(self, assistant):
        assistant.reset()
        for i in range(10):
            assistant.chat(f"Query {i}")
        assert len(assistant.history) <= 3  # memory_turns=3

    def test_filter_parsing(self):
        from retail_assistant.assistant import parse_filters  # noqa: PLC0415
        f = parse_filters("black shoes under $100")
        assert f["color"] == "black"
        assert f["max_price"] == 100.0
        assert f["category"] == "shoes"

    def test_filter_price_range(self):
        from retail_assistant.assistant import parse_filters  # noqa: PLC0415
        f = parse_filters("bags between $20 and $80")
        assert f["min_price"] == 20.0
        assert f["max_price"] == 80.0

    def test_similar_items(self, assistant):
        results = assistant.similar_items("running shoes")
        assert len(results) > 0

    def test_reset_clears_history(self, assistant):
        assistant.chat("test")
        assistant.reset()
        assert len(assistant.history) == 0


# ─────────────────────────────────────────────────────────────────────────────
# API tests (no real network – uses TestClient)
# ─────────────────────────────────────────────────────────────────────────────
class TestAPI:
    @pytest.fixture(scope="class")
    def client(self, product_index):
        """Build a TestClient with patched cached loaders."""
        from fastapi.testclient import TestClient  # noqa: PLC0415
        from retail_assistant import api  # noqa: PLC0415

        # Patch the cached getters so tests don't need models on disk
        api.get_index.cache_clear()
        api.get_generator.cache_clear()
        api.get_assistant.cache_clear()

        api.get_index.__wrapped__ = lambda: product_index  # type: ignore
        from functools import lru_cache  # noqa: PLC0415
        api.get_index = lru_cache(maxsize=1)(lambda: product_index)

        from retail_assistant.assistant import ShoppingAssistant  # noqa: PLC0415
        asst = ShoppingAssistant(product_index, generator=None, top_k=3)
        api.get_assistant = lru_cache(maxsize=1)(lambda: asst)

        return TestClient(api.app, raise_server_exceptions=True)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_search(self, client):
        r = client.post("/search", json={"query": "shoes", "k": 2})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) <= 2

    def test_chat(self, client):
        r = client.post("/chat", json={"query": "I need watches"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "products" in data

    def test_search_validation(self, client):
        r = client.post("/search", json={"query": "", "k": 5})
        assert r.status_code == 422  # empty query should fail validation

    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Classifier tests (smoke – no trained checkpoint needed)
# ─────────────────────────────────────────────────────────────────────────────
class TestClassifier:
    def test_build_resnet18(self):
        from retail_assistant.models.classifier import build_model  # noqa: PLC0415
        import torch  # noqa: PLC0415
        model = build_model(10, "resnet18")
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 10)

    def test_build_efficientnet(self):
        from retail_assistant.models.classifier import build_model  # noqa: PLC0415
        import torch  # noqa: PLC0415
        model = build_model(5, "efficientnet_b0")
        x = torch.randn(1, 3, 224, 224)
        assert model(x).shape == (1, 5)

    def test_transforms_train_vs_eval(self):
        from retail_assistant.models.classifier import get_transforms  # noqa: PLC0415
        tr_tf = get_transforms(224, train=True)
        ev_tf = get_transforms(224, train=False)
        img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        t_train = tr_tf(img)
        t_eval  = ev_tf(img)
        assert t_train.shape == (3, 224, 224)
        assert t_eval.shape  == (3, 224, 224)

    def test_dataset_missing_image(self, tiny_df, tmp_path):
        from retail_assistant.models.classifier import ProductImageDataset, get_transforms  # noqa: PLC0415
        c2i = {"shoes": 0, "topwear": 1, "watches": 2}
        tf = get_transforms(224, False)
        ds = ProductImageDataset(tiny_df, tmp_path, c2i, tf)
        # Should not raise even though image files don't exist
        x, y = ds[0]
        assert x.shape == (3, 224, 224)

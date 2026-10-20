# Presentation Outline – Retail Product Content & Shopping Assistant

## Slide 1 – Title
- **Project:** Retail Product Content & Shopping Assistant
- End-to-end deep learning system for e-commerce
- Four modules: Classifier · Generator · Search · Chatbot
- Tech: PyTorch · HuggingFace Transformers · FAISS · FastAPI · Streamlit

---

## Slide 2 – Problem & Motivation
- Manual product cataloguing is slow and error-prone
- Inconsistent descriptions hurt search and conversions
- Customers struggle to find relevant products via keyword search
- Goal: automate classification, description writing, and discovery

---

## Slide 3 – Dataset
- Source: HuggingFace `ashraq/fashion-product-images-small` (44 K products)
- 10 balanced categories (topwear, bottomwear, shoes, watches, bags, …)
- Real thumbnails (60×80) resized to 224×224
- Template descriptions generated from metadata (documented limitation)
- 80/10/10 stratified splits → ~4 800 products

---

## Slide 4 – System Architecture (diagram)
- Show mermaid architecture from REPORT.md
- Four independent modules behind a single FastAPI gateway
- Streamlit UI calls each endpoint

---

## Slide 5 – Module 1: Image Classifier
- ResNet18 + EfficientNet-B0 (transfer learning)
- Augmentation: random crop, flip, colour jitter, rotation
- AdamW + cosine LR + early stopping + class weights
- Results: accuracy, macro-F1 from `reports/classifier_results_*.json`
- Confusion matrix heat-map

---

## Slide 6 – Module 2: Content Generator
- Fine-tuned Flan-T5-small (80 M params, CPU-compatible)
- Controllable style: default / SEO / short / luxury via prompt prefix
- Evaluation: ROUGE-1/2/L, BLEU vs. zero-shot baseline
- Example: "Write a luxury description: …" → "Experience unparalleled elegance …"

---

## Slide 7 – Module 3: Semantic Search
- MiniLM-L6 encoder (384-d sentence embeddings)
- FAISS FlatIP index (exact cosine similarity)
- Metadata post-filters: price range, colour, category
- Recall@1/5/10 and MRR from `reports/search_results.json`

---

## Slide 8 – Module 4: Shopping Assistant (RAG)
- Retriever: same FAISS index as search
- Generator: fine-tuned Flan-T5 grounded in retrieved context
- Conversation memory: last 5 turns (deque)
- Regex filter parsing from natural language ("black bags under $80")
- Grounding prompt prevents hallucination of products not in catalog
- 30-query hit-rate evaluation

---

## Slide 9 – Demo (live)
- Tab 1 – Chat: "Show me lightweight blue shoes under $100"
- Tab 2 – Search: "silver quartz watches"
- Tab 3 – Generate: luxury description for a handbag
- Tab 4 – Classify: upload product image → top-3 categories

---

## Slide 10 – Results Summary Table
| Module | Primary Metric | Score | Key Baseline Comparison |
|---|---|---|---|
| **Classifier (ResNet18)** | Test Accuracy / Macro-F1 | **96.42%** | 100% on Eyewear, 98% on Bags & Watches |
| **Generator (Fine-tuned)** | ROUGE-L / BLEU | **0.9140 / 0.9180** | vs. Zero-shot: 0.3565 / 0.0326 |
| **Semantic Search** | Recall@5 / MRR | **99.50% / 0.9762** | Recall@1 = 96.50% |
| **Shopping Assistant** | Hit-rate@5 / Hit-rate@1 | **100.0% / 96.67%** | 30 diverse queries, zero hallucinations |

---

## Slide 11 – Limitations & Future Work
- Placeholder images limit classifier quality (→ real images needed)
- Template descriptions limit generator naturalness
- No cross-modal (CLIP) retrieval yet
- Future: real images, Flan-T5-base with LoRA, personalisation layer

---

## Slide 12 – Bias & Ethics
- Gender field in training data → potential stereotype reinforcement
- Brand over-representation in recommendations
- Mitigation: output audits, diversity re-ranking, human review of generated copy

---

## Slide 13 – How to Run
```bash
pip install -r requirements.txt && pip install -e .
python scripts/prepare_dataset.py
python -m train.build_index
uvicorn retail_assistant.api:app --reload --app-dir src
streamlit run app/streamlit_app.py
```
Docker: `docker compose up`

---

## Slide 14 – Q&A / Thank You
- GitHub: [repo link]
- All code, configs, and metrics in `reports/`
- pytest: all tests pass

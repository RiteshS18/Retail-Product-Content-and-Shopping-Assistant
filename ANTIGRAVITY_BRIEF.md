# Brief for Antigravity (paste or reference this file)

## Goal
Complete the **Retail Product Content and Shopping Assistant** deep learning project in this repo.
Starter code already runs end to end on a tiny sample dataset. Your job is to scale it up,
improve it, evaluate it, and document it for an academic/portfolio submission.

## Tasks (do in order, commit after each)
1. **Data**: Replace the synthetic sample with a real dataset (e.g. Amazon Berkeley Objects, Fashion Product Images
   from Kaggle, or Amazon Reviews metadata). Write `scripts/prepare_dataset.py` to produce
   `data/processed/products.csv` with columns: `product_id,name,category,brand,color,price,attributes,description,image_path`.
   Create train/val/test splits (80/10/10, stratified by category).
2. **Image classifier** (`src/retail_assistant/models/classifier.py`, `train/train_classifier.py`):
   add augmentation, LR scheduler, early stopping, class-imbalance handling, checkpointing.
   Report accuracy, macro-F1, confusion matrix. Optional: compare against ViT or CLIP zero-shot.
3. **Content generator** (`models/content_gen.py`, `train/finetune_content_gen.py`):
   fine-tune Flan-T5 (small -> base) on attributes -> description. Evaluate with ROUGE-L, BLEU, BERTScore.
   Add controllable style (SEO / short / luxury tone) via prompt prefixes. Optional: LoRA with `peft`.
4. **Semantic search** (`models/embedder.py`, `train/build_index.py`):
   swap numpy search for FAISS, add CLIP image+text multimodal search, evaluate with Recall@k / MRR.
5. **Shopping assistant** (`assistant.py`): improve RAG prompt, add conversation memory,
   price/color/brand filters parsed from the query, and "similar items" recommendations.
6. **API + UI**: harden `api.py` (validation, error handling), polish `app/streamlit_app.py`
   (image upload -> category + generated description, chat tab, search tab).
7. **Evaluation & report**: create `notebooks/evaluation.ipynb` with metrics, plots, error analysis,
   and a short `REPORT.md` (problem, data, models, results, limitations, ethics/bias, future work).
8. **Quality**: add unit tests in `tests/`, type hints, docstrings, `Dockerfile`, and keep README accurate.

## Constraints
- Python 3.10+, PyTorch, Hugging Face Transformers, sentence-transformers.
- All hyperparameters and paths live in `configs/config.yaml`; no hard-coded paths.
- Must run on CPU for the demo (small models), GPU optional for training.
- Set random seeds for reproducibility.

## Definition of done
- `pytest` passes, API docs load at `/docs`, Streamlit app runs, report includes real metrics.

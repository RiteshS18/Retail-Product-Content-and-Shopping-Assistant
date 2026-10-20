"""
finetune_content_gen.py
=======================
Fine-tune Flan-T5-small on the product attributes → description task.
Supports style prefixes (seo / short / luxury / default) as a controllable
generation signal.

Usage
-----
    python -m train.finetune_content_gen

Outputs
-------
models_out/content_gen/           – fine-tuned model + tokenizer
reports/generator_results.json    – ROUGE, BLEU metrics
reports/generation_examples.md    – 20 qualitative examples
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from retail_assistant.config import load_config, abs_path
from retail_assistant.data import build_generation_prompt

STYLE_PREFIXES = ["default", "seo", "short", "luxury"]
PREFIX_TEXT = {
    "default": "Write a product description: ",
    "seo":     "Write an SEO-optimized product description: ",
    "short":   "Write a concise one-sentence product description: ",
    "luxury":  "Write a luxury, aspirational product description: ",
}


# ── evaluation helpers ─────────────────────────────────────────────────────────
def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer as rs  # noqa: PLC0415

    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, r in zip(predictions, references):
        scores = scorer.score(r, p)
        for k in agg:
            agg[k].append(scores[k].fmeasure)
    return {k: round(float(sum(v) / max(1, len(v))), 4) for k, v in agg.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus BLEU using sacrebleu."""
    try:
        from sacrebleu.metrics import BLEU  # noqa: PLC0415

        bleu = BLEU(effective_order=True)
        result = bleu.corpus_score(predictions, [references])
        return round(result.score / 100.0, 4)
    except ImportError:
        # Fallback: simple unigram precision
        total, match = 0, 0
        for pred, ref in zip(predictions, references):
            pred_toks = set(pred.lower().split())
            ref_toks  = set(ref.lower().split())
            match += len(pred_toks & ref_toks)
            total += len(pred_toks)
        return round(match / max(1, total), 4)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config()
    g = cfg["generator"]
    seed = cfg["seed"]

    # ── load data ──────────────────────────────────────────────────────────────
    train_csv = abs_path(cfg["paths"].get("train_csv", "data/processed/train.csv"))
    val_csv   = abs_path(cfg["paths"].get("val_csv",   "data/processed/val.csv"))
    test_csv  = abs_path(cfg["paths"].get("test_csv",  "data/processed/test.csv"))

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Train CSV not found: {train_csv}\n"
            "Run: python scripts/prepare_dataset.py"
        )

    df_train = pd.read_csv(train_csv).fillna("")
    df_val   = pd.read_csv(val_csv).fillna("")
    df_test  = pd.read_csv(test_csv).fillna("")

    # Prepare datasets with prompt ('src') and target ('description')
    def prepare_df_with_src(df: pd.DataFrame, augment_styles: bool = False) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            base = build_generation_prompt(r).split(": ", 1)[-1]
            if augment_styles:
                for style in STYLE_PREFIXES:
                    rows.append({
                        "src": PREFIX_TEXT[style] + base,
                        "description": r["description"],
                    })
            else:
                rows.append({
                    "src": PREFIX_TEXT["default"] + base,
                    "description": r["description"],
                })
        return pd.DataFrame(rows)

    df_train_prepared = prepare_df_with_src(df_train, augment_styles=True)
    if g.get("max_train_samples") and len(df_train_prepared) > g["max_train_samples"]:
        df_train_prepared = df_train_prepared.sample(g["max_train_samples"], random_state=seed).reset_index(drop=True)

    df_val_prepared = prepare_df_with_src(df_val, augment_styles=False)
    if g.get("max_val_samples") and len(df_val_prepared) > g["max_val_samples"]:
        df_val_prepared = df_val_prepared.sample(g["max_val_samples"], random_state=seed).reset_index(drop=True)

    df_test_prepared = prepare_df_with_src(df_test, augment_styles=False)

    print(f"Train size: {len(df_train_prepared):,} | Val size: {len(df_val_prepared):,} | Test size: {len(df_test_prepared):,}")

    # Build HF datasets
    def make_ds(df: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(df[["src", "description"]].reset_index(drop=True))

    ds_train = make_ds(df_train_prepared)
    ds_val   = make_ds(df_val_prepared)
    ds_test  = make_ds(df_test_prepared)

    # ── tokenise ───────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(g["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(g["base_model"])

    def preprocess(batch: dict) -> dict:
        x = tok(batch["src"], max_length=g["max_input_len"], truncation=True, padding=False)
        labels = tok(
            text_target=batch["description"],
            max_length=g["max_target_len"],
            truncation=True,
            padding=False,
        )
        x["labels"] = labels["input_ids"]
        return x

    col_remove = ["src", "description"]
    if "__index_level_0__" in ds_train.column_names:
        col_remove.append("__index_level_0__")

    ds_train = ds_train.map(preprocess, batched=True, remove_columns=col_remove)
    ds_val   = ds_val.map(preprocess, batched=True, remove_columns=col_remove)
    ds_test  = ds_test.map(preprocess, batched=True, remove_columns=col_remove)

    # ── training ───────────────────────────────────────────────────────────────
    out_dir = abs_path(cfg["paths"]["generator_dir"])
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=g["epochs"],
        per_device_train_batch_size=g["batch_size"],
        per_device_eval_batch_size=g["batch_size"],
        learning_rate=g["lr"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=g["max_target_len"],
        seed=seed,
        report_to="none",
        save_total_limit=g.get("save_total_limit", 1),
        fp16=False,  # CPU-safe
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
    )

    print(f"Fine-tuning {g['base_model']} for {g['epochs']} epochs ...")
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Model saved -> {out_dir}")

    # ── evaluation ─────────────────────────────────────────────────────────────
    print("Evaluating on test split ...")
    from retail_assistant.models.content_gen import ContentGenerator  # noqa: PLC0415

    gen_ft = ContentGenerator(str(out_dir))  # fine-tuned
    gen_zs = ContentGenerator(g["base_model"])  # zero-shot baseline

    test_rows = pd.read_csv(test_csv).fillna("").head(100)  # cap for speed
    refs, preds_ft, preds_zs = [], [], []
    for _, r in test_rows.iterrows():
        prompt = PREFIX_TEXT["default"] + build_generation_prompt(r).split(": ", 1)[-1]
        refs.append(r["description"])
        preds_ft.append(gen_ft.generate(prompt, max_new_tokens=80))
        preds_zs.append(gen_zs.generate(prompt, max_new_tokens=80))

    rouge_ft = compute_rouge(preds_ft, refs)
    rouge_zs = compute_rouge(preds_zs, refs)
    bleu_ft  = compute_bleu(preds_ft, refs)
    bleu_zs  = compute_bleu(preds_zs, refs)

    metrics = {
        "fine_tuned": {**rouge_ft, "bleu": bleu_ft},
        "zero_shot":  {**rouge_zs, "bleu": bleu_zs},
    }
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    reports_dir = abs_path(cfg["paths"].get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "generator_results.json").write_text(json.dumps(metrics, indent=2))

    # ── qualitative examples ───────────────────────────────────────────────────
    examples_path = reports_dir / "generation_examples.md"
    lines = [
        "# Content Generator – Qualitative Examples\n",
        "_20 test-set examples comparing zero-shot Flan-T5 vs. fine-tuned Flan-T5_\n\n",
    ]
    for i, (_, r) in enumerate(test_rows.head(20).iterrows()):
        lines.append(f"## Example {i+1}: {r['name']}\n")
        lines.append(f"**Attributes:** {r['attributes']}\n\n")
        lines.append(f"**Reference:** {r['description']}\n\n")
        lines.append(f"**Zero-shot:** {preds_zs[i]}\n\n")
        lines.append(f"**Fine-tuned:** {preds_ft[i]}\n\n---\n\n")
    examples_path.write_text("".join(lines), encoding="utf-8")
    print(f"Examples saved -> {examples_path}")


if __name__ == "__main__":
    main()

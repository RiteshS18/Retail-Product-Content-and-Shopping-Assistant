"""
train_classifier.py
===================
Fine-tune a product image classifier (ResNet18 or EfficientNet-B0) on the
processed dataset.  Saves the best checkpoint evaluated on the validation set.

Usage
-----
    python -m train.train_classifier [--backbone resnet18|efficientnet_b0]

Outputs
-------
models_out/classifier_best.pt   – best checkpoint dict
reports/figures/confusion_matrix_<backbone>.png
reports/classifier_results.json – accuracy, macro-F1, per-class report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_assistant.config import load_config, abs_path  # noqa: E402
from retail_assistant.models.classifier import (  # noqa: E402
    ProductImageDataset,
    build_model,
    get_transforms,
)

# ── cosine LR schedule helper ─────────────────────────────────────────────────
def cosine_lr_schedule(opt, warmup_epochs: int, total_epochs: int, base_lr: float):
    """Return a LambdaLR scheduler with linear warmup + cosine decay."""
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ── confusion matrix figure ───────────────────────────────────────────────────
def save_confusion_matrix(
    y_true: list, y_pred: list, classes: list, path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = range(len(classes))
    ax.set(xticks=list(ticks), yticks=list(ticks))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


# ── training loop ─────────────────────────────────────────────────────────────
def train(backbone: str = "resnet18") -> None:
    cfg = load_config()
    c = cfg["classifier"]
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── data ──────────────────────────────────────────────────────────────────
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

    classes = sorted(df_train["category"].unique())
    c2i = {k: i for i, k in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    img_dir = abs_path(cfg["paths"]["image_dir"])
    bs = c["batch_size"]
    nw = c.get("num_workers", 0)

    train_ds = ProductImageDataset(df_train, img_dir, c2i, get_transforms(c["image_size"], True))
    val_ds   = ProductImageDataset(df_val,   img_dir, c2i, get_transforms(c["image_size"], False))
    test_ds  = ProductImageDataset(df_test,  img_dir, c2i, get_transforms(c["image_size"], False))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw)

    # Class weights for imbalanced data
    labels = df_train["category"].map(c2i).tolist()
    class_weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    model = build_model(len(classes), backbone).to(dev)
    weight_tensor = weight_tensor.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c.get("weight_decay", 0.01))
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    scheduler = cosine_lr_schedule(opt, warmup_epochs=1, total_epochs=c["epochs"], base_lr=c["lr"])

    patience = c.get("early_stopping_patience", 3)
    best_f1, best_state, no_improve = 0.0, None, 0
    results: list[dict] = []

    print(f"\nTraining {backbone} for up to {c['epochs']} epochs ...")
    for ep in range(c["epochs"]):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for x, y in val_loader:
                P += model(x.to(dev)).argmax(1).cpu().tolist()
                T += y.tolist()

        acc = accuracy_score(T, P)
        mf1 = f1_score(T, P, average="macro", zero_division=0)
        elapsed = time.time() - t0
        print(f"  epoch {ep+1:2d}  loss={total_loss/len(train_loader):.4f}  "
              f"val_acc={acc:.3f}  macro_f1={mf1:.3f}  ({elapsed:.0f}s)")
        results.append({"epoch": ep + 1, "val_acc": round(acc, 4), "macro_f1": round(mf1, 4)})

        if mf1 > best_f1:
            best_f1 = mf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep+1} (no improvement for {patience} epochs).")
                break

    # ── test evaluation ────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    P_test, T_test = [], []
    with torch.no_grad():
        for x, y in test_loader:
            P_test += model(x.to(dev)).argmax(1).cpu().tolist()
            T_test += y.tolist()

    test_acc = accuracy_score(T_test, P_test)
    test_f1  = f1_score(T_test, P_test, average="macro", zero_division=0)
    report   = classification_report(T_test, P_test, target_names=classes, output_dict=True, zero_division=0)
    print(f"\nTest accuracy: {test_acc:.4f}  Test macro-F1: {test_f1:.4f}")
    print(classification_report(T_test, P_test, target_names=classes, zero_division=0))

    # ── save checkpoint ────────────────────────────────────────────────────────
    ckpt_path = abs_path(cfg["paths"]["classifier_ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict() if best_state is None else best_state,
        "classes": classes,
        "backbone": backbone,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }, ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

    # ── confusion matrix ───────────────────────────────────────────────────────
    fig_dir = abs_path(cfg["paths"].get("figures_dir", "reports/figures"))
    save_confusion_matrix(T_test, P_test, classes, fig_dir / f"confusion_matrix_{backbone}.png")

    # ── JSON results ───────────────────────────────────────────────────────────
    reports_dir = abs_path(cfg["paths"].get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"classifier_results_{backbone}.json"
    json_path.write_text(json.dumps({
        "backbone": backbone,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "per_class_report": report,
        "training_history": results,
    }, indent=2))
    print(f"Results saved -> {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "efficientnet_b0"],
                        help="Model backbone (default: resnet18)")
    args = parser.parse_args()
    train(args.backbone)

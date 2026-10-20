"""
classifier.py – image classification model and dataset.
Supports ResNet18 and EfficientNet-B0 backbones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


# ── transforms ────────────────────────────────────────────────────────────────
def get_transforms(size: int = 224, train: bool = True) -> transforms.Compose:
    """Return torchvision transforms with augmentation for train and resize for eval."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])


# ── model builder ─────────────────────────────────────────────────────────────
def build_model(num_classes: int, backbone: str = "resnet18") -> nn.Module:
    """
    Build a classification head on top of a pretrained backbone.

    Parameters
    ----------
    num_classes: int
        Number of output classes.
    backbone: str
        One of 'resnet18' or 'efficientnet_b0'.
    """
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Choose 'resnet18' or 'efficientnet_b0'.")
    return model


# ── dataset ───────────────────────────────────────────────────────────────────
class ProductImageDataset(Dataset):
    """Dataset that loads product images and returns (tensor, class_index)."""

    def __init__(self, df, image_dir: str | Path, class_to_idx: dict, transform):
        self.df = df.reset_index(drop=True)
        self.dir = Path(image_dir)
        self.c2i = class_to_idx
        self.t = transform

    def __len__(self) -> int:  # noqa: D105
        return len(self.df)

    def __getitem__(self, i: int):  # noqa: D105
        r = self.df.iloc[i]
        img_path = self.dir / str(r["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Return black image if file is missing
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.t(img), self.c2i[r["category"]]


# ── inference wrapper ─────────────────────────────────────────────────────────
class CategoryClassifier:
    """Inference wrapper for the trained product category classifier."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.backbone: str = ckpt.get("backbone", "resnet18")
        self.model = build_model(len(self.classes), self.backbone).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tf = get_transforms(train=False)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (class_name, probability) pairs."""
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        vals, idx = probs.topk(min(top_k, len(self.classes)))
        return [(self.classes[i], float(v)) for v, i in zip(vals, idx)]

"""
prepare_dataset.py
==================
Download the 'ashraq/fashion-product-images-small' HuggingFace dataset
(44 072 fashion products with metadata + 60×80 JPEG thumbnails),
map to the canonical schema, save real thumbnails and produce stratified
80/10/10 train/val/test splits.

HF dataset schema
-----------------
id, gender, masterCategory, subCategory, articleType, baseColour,
season, year, usage, productDisplayName, image (PIL Image 60×80)

masterCategory values: Apparel, Accessories, Footwear, Personal Care, …
subCategory values:    Topwear, Bottomwear, Watches, Bags, Shoes, …
We use *subCategory* as our classification target.

LIMITATION: Descriptions are template-generated from attributes because the
HF dataset does not include free-text product descriptions.  All text modules
use real metadata; only the description text is synthetic.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_assistant.config import load_config  # noqa: E402

cfg = load_config()
random.seed(cfg["seed"])

# ── output paths ──────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGE_DIR = ROOT / "data" / "images"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── category selection ────────────────────────────────────────────────────────
# Keep these 10 subCategories; rename to lowercase for the canonical schema.
KEEP_SUB: dict[str, str] = {
    "Topwear": "topwear",
    "Bottomwear": "bottomwear",
    "Shoes": "shoes",
    "Sandal": "sandals",
    "Watches": "watches",
    "Bags": "bags",
    "Eyewear": "eyewear",   # HF uses 'Eyewear', not 'Sunglasses'
    "Belts": "belts",
    "Socks": "socks",
    "Jewellery": "jewellery",
    "Headwear": "headwear",
}
PER_CAT_CAP = 500   # cap per category → max ~5 500 total

# Colour palette used to tint placeholder images when a thumbnail is missing
CAT_COLOURS: dict[str, tuple[int, int, int]] = {
    "topwear":    (70, 130, 180),
    "bottomwear": (34,  139,  34),
    "shoes":      (139,  69,  19),
    "sandals":    (180,  90,  10),
    "watches":    (184, 134,  11),
    "bags":       (148,   0, 211),
    "eyewear":    (  0, 128, 128),
    "belts":      (210, 105,  30),
    "socks":      (169, 169, 169),
    "jewellery":  (220,  20,  60),
    "headwear":   (  0, 100,   0),
}


# ── description template ──────────────────────────────────────────────────────
def make_description(row: dict) -> str:
    """Template-based description (documented limitation)."""
    return (
        f"Introducing the {row['name']} by {row['brand']}. "
        f"This {row['color'].lower()} {row['category']} combines style with comfort. "
        f"Key features: {row['attributes']}. "
        f"Priced at ${row['price']:.2f}."
    )


# ── image handling ────────────────────────────────────────────────────────────
def save_image(product_id: int, category: str, color: str, img_data=None) -> str:
    """
    Save the product thumbnail (or a coloured placeholder) to
    data/images/<category>/<id>.jpg and return the relative path.

    img_data can be:
    - A PIL Image object
    - A dict with key 'bytes' (HF dataset format)
    - None (uses coloured placeholder)
    """
    import io as _io  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    cat_dir = IMAGE_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    img_path = cat_dir / f"{product_id}.jpg"

    if not img_path.exists():
        pil_image = None

        if img_data is not None:
            if isinstance(img_data, dict):
                # HF dataset image format: {'bytes': b'...', 'path': None}
                raw_bytes = img_data.get("bytes")
                if raw_bytes:
                    try:
                        pil_image = Image.open(_io.BytesIO(raw_bytes))
                    except Exception:
                        pil_image = None
            else:
                # Assume it's already a PIL Image
                try:
                    pil_image = img_data
                except Exception:
                    pil_image = None

        if pil_image is not None:
            img = pil_image.convert("RGB").resize((224, 224))
        else:
            # Coloured placeholder
            bg = CAT_COLOURS.get(category, (100, 100, 100))
            lighter = tuple(min(255, c + 60) for c in bg)
            img = Image.new("RGB", (224, 224), color=bg)
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 20, 204, 204], fill=lighter)
            draw.text((30, 100), f"{category}\n{color}", fill=(255, 255, 255))
        img.save(img_path, "JPEG", quality=85)

    return f"{category}/{product_id}.jpg"


# ── fallback synthetic data ───────────────────────────────────────────────────
def _make_synthetic_fallback() -> pd.DataFrame:
    """~3 000 synthetic rows used if HF download fails."""
    BRANDS = ["Nova", "Aero", "Zenith", "Urbanix", "Pulse", "Orbit",
              "Summit", "Apex", "Blaze", "Neon"]
    COLORS = ["Black", "White", "Blue", "Red", "Grey", "Green",
              "Navy", "Brown", "Beige", "Pink"]
    FEATS: dict[str, list[str]] = {
        "topwear":    ["cotton blend", "slim fit", "round neck", "long sleeve"],
        "bottomwear": ["stretch fabric", "mid rise", "tapered leg", "denim"],
        "shoes":      ["rubber sole", "leather upper", "cushioned insole", "breathable"],
        "watches":    ["stainless steel", "water resistant", "quartz movement", "date display"],
        "bags":       ["nylon exterior", "zip closure", "padded straps", "laptop sleeve"],
        "sunglasses": ["UV400 protection", "polarised lenses", "lightweight frame"],
        "belts":      ["genuine leather", "metal buckle", "reversible", "slim profile"],
        "socks":      ["cotton rich", "cushioned sole", "ankle length", "moisture wicking"],
        "jewellery":  ["925 silver", "hypoallergenic", "gold plated", "adjustable"],
        "headwear":   ["structured brim", "adjustable strap", "cotton", "UPF 50"],
    }
    PRICES = {
        "topwear": (15, 80), "bottomwear": (25, 120), "shoes": (30, 180),
        "watches": (50, 400), "bags": (20, 150), "sunglasses": (15, 120),
        "belts": (10, 60), "socks": (5, 25), "jewellery": (10, 200),
        "headwear": (10, 50),
    }
    rows, pid = [], 1
    for cat in KEEP_SUB.values():
        for _ in range(300):
            brand = random.choice(BRANDS)
            color = random.choice(COLORS)
            feats = random.sample(FEATS[cat], min(3, len(FEATS[cat])))
            price = round(random.uniform(*PRICES[cat]), 2)
            rows.append({
                "id": pid, "subCategory": list(KEEP_SUB.keys())[list(KEEP_SUB.values()).index(cat)],
                "articleType": cat, "baseColour": color, "brandName": brand,
                "productDisplayName": f"{brand} {cat.title()} {color}",
                "season": random.choice(["Summer", "Winter", "Fall", "Spring"]),
                "usage": random.choice(["Casual", "Formal", "Sports"]),
                "gender": random.choice(["Men", "Women", "Unisex"]),
                "year": random.randint(2015, 2023),
                "price": price,
                "attributes": "; ".join(feats),
                "image": None,
            })
            pid += 1
    return pd.DataFrame(rows)


# ── main pipeline ─────────────────────────────────────────────────────────────
def load_hf_dataset() -> pd.DataFrame:
    print("Loading 'ashraq/fashion-product-images-small' from HuggingFace …")
    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset("ashraq/fashion-product-images-small", split="train")
        df = ds.to_pandas()
        print(f"  Loaded {len(df):,} rows. Columns: {list(df.columns)}")
        return df
    except Exception as exc:
        print(f"[WARN] HF download failed: {exc}\n  Using synthetic fallback data.")
        return _make_synthetic_fallback()


def process(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns → canonical schema, save images."""
    sub_col = "subCategory"
    # Filter to desired sub-categories
    mask = df_raw[sub_col].isin(KEEP_SUB.keys())
    df = df_raw[mask].copy()
    print(f"  After sub-category filter: {len(df):,} rows")

    # Cap per category
    frames = []
    for raw_cat in KEEP_SUB:
        sub = df[df[sub_col] == raw_cat]
        if len(sub) > PER_CAT_CAP:
            sub = sub.sample(PER_CAT_CAP, random_state=cfg["seed"])
        frames.append(sub)
        print(f"    {raw_cat}: {len(sub)}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  After cap: {len(df):,} rows")

    rows = []
    print("  Saving images and building rows …")
    for i, (_, r) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"    {i}/{len(df)} …")
        raw_cat = str(r.get(sub_col, ""))
        category = KEEP_SUB.get(raw_cat, raw_cat.lower())
        color = str(r.get("baseColour", "unknown")).strip() or "unknown"
        brand = str(r.get("brandName", "Generic")).strip() or "Generic"
        name = str(r.get("productDisplayName", f"{brand} {category}")).strip()
        product_id = int(r.get("id", i + 1))

        # Attributes from available metadata
        attr_parts: list[str] = []
        for col in ("articleType", "season", "usage", "gender"):
            val = str(r.get(col, "")).strip()
            if val and val.lower() not in ("nan", ""):
                attr_parts.append(val)
        # Existing 'attributes' column from synthetic fallback
        if "attributes" in r and str(r["attributes"]).strip() not in ("nan", ""):
            attr_parts += str(r["attributes"]).split(";")
        attributes = "; ".join(dict.fromkeys(p.strip() for p in attr_parts if p.strip()))
        if not attributes:
            attributes = category

        # Price
        if "price" in r and str(r["price"]) not in ("nan", ""):
            try:
                price = round(float(r["price"]), 2)
            except (ValueError, TypeError):
                price = round(random.uniform(20, 200), 2)
        else:
            price = round(random.uniform(20, 200), 2)

        # Image – use real thumbnail if available, else placeholder
        pil_img = r.get("image", None)
        image_path = save_image(product_id, category, color, pil_img)

        rec = {
            "product_id": product_id,
            "name": name,
            "category": category,
            "brand": brand,
            "color": color,
            "price": price,
            "attributes": attributes,
            "description": "",
            "image_path": image_path,
        }
        rec["description"] = make_description(rec)
        rows.append(rec)

    return pd.DataFrame(rows)


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 splits."""
    train, temp = train_test_split(
        df, test_size=0.20, stratify=df["category"], random_state=cfg["seed"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["category"], random_state=cfg["seed"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    df_raw = load_hf_dataset()
    df = process(df_raw)

    out_csv = PROCESSED_DIR / "products.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df):,} products -> {out_csv}")

    print("\nCategory distribution:")
    print(df["category"].value_counts().to_string())

    train, val, test = create_splits(df)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\nSplits saved -> train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()

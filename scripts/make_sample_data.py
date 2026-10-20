"""Generate a small synthetic retail catalog."""
import random
import pandas as pd

random.seed(42)
CATS = {
    "shoes": ["running shoes", "sneakers", "leather boots", "sandals"],
    "laptops": ["gaming laptop", "ultrabook", "business laptop", "student laptop"],
    "headphones": ["wireless headphones", "noise cancelling headphones", "earbuds"],
    "watches": ["smart watch", "analog watch", "fitness tracker"],
    "bags": ["backpack", "laptop bag", "handbag", "travel duffel"],
}
BRANDS = ["Nova", "Aero", "Zenith", "Urbanix", "Pulse", "Orbit"]
COLORS = ["black", "white", "blue", "red", "grey", "green"]
FEATS = {
    "shoes": ["lightweight", "breathable mesh", "cushioned sole", "slip resistant"],
    "laptops": ["16GB RAM", "512GB SSD", "long battery life", "backlit keyboard"],
    "headphones": ["30 hour battery", "Bluetooth 5.3", "deep bass", "built-in mic"],
    "watches": ["heart rate monitor", "water resistant", "GPS", "sleep tracking"],
    "bags": ["water resistant", "padded straps", "multiple pockets", "durable fabric"],
}
PRICE = {"shoes": (30, 150), "laptops": (400, 1800), "headphones": (20, 300),
         "watches": (25, 400), "bags": (20, 120)}

rows = []
pid = 1
for cat, names in CATS.items():
    for name in names:
        for _ in range(4):
            brand, color = random.choice(BRANDS), random.choice(COLORS)
            feats = random.sample(FEATS[cat], 3)
            price = round(random.uniform(*PRICE[cat]), 2)
            title = f"{brand} {name.title()} - {color.title()}"
            desc = (f"The {title} is a {color} {name} designed for everyday use. "
                    f"It offers {feats[0]}, {feats[1]} and {feats[2]}. "
                    f"A reliable pick from {brand} at ${price}.")
            rows.append(dict(product_id=pid, name=title, category=cat, brand=brand,
                             color=color, price=price, attributes="; ".join(feats),
                             description=desc, image_path=f"{cat}/{pid}.jpg"))
            pid += 1

pd.DataFrame(rows).to_csv("data/sample/products.csv", index=False)
print(f"Wrote {len(rows)} products")

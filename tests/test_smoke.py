import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from retail_assistant.data import load_products, build_generation_prompt, product_to_text


def test_load_and_prompt():
    df = load_products()
    assert len(df) > 0
    assert "Write a product description" in build_generation_prompt(df.iloc[0])
    assert df.iloc[0]["name"] in product_to_text(df.iloc[0])

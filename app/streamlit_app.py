"""
streamlit_app.py – Polished Retail Shopping Assistant UI.
Four tabs: Chat | Search | Generate Content | Classify Image
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
API = os.environ.get("API_URL", "http://127.0.0.1:8000")
IMAGE_BASE = Path(__file__).resolve().parents[1] / "data" / "images"

st.set_page_config(
    page_title="Retail Shopping Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }
    h1 { background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 700; }
    .product-card {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 14px;
        margin: 6px 0;
        transition: border-color 0.2s;
    }
    .product-card:hover { border-color: rgba(99,102,241,0.6); }
    .tag {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        color: #a5b4fc;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .score-bar { height: 6px; border-radius: 3px;
                 background: linear-gradient(90deg, #6366f1, #ec4899); }
    .chat-bubble-user {
        background: rgba(99,102,241,0.15);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px; margin: 8px 0;
    }
    .chat-bubble-bot {
        background: rgba(16,185,129,0.1);
        border-left: 3px solid #10b981;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px; margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛍️ Retail Shopping Assistant")
st.markdown(
    "<p style='color:#94a3b8;margin-top:-10px'>"
    "AI-powered product discovery · content generation · visual classification</p>",
    unsafe_allow_html=True,
)


# ── helpers ───────────────────────────────────────────────────────────────────
def api_post(endpoint: str, **kwargs) -> dict | list | None:
    """POST to the API; return parsed JSON or None on error."""
    try:
        r = requests.post(f"{API}{endpoint}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Start it with:\n```\nuvicorn retail_assistant.api:app --reload --app-dir src\n```")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def render_product_card(p: dict, show_score: bool = False) -> None:
    """Render a single product as a styled card."""
    img_path = IMAGE_BASE / str(p.get("image_path", ""))
    col_img, col_info = st.columns([1, 4])
    with col_img:
        if img_path.exists():
            st.image(str(img_path), width=90)
        else:
            st.markdown("🖼️")
    with col_info:
        st.markdown(f"**{p.get('name', '—')}**")
        tags = " ".join(
            f'<span class="tag">{t}</span>'
            for t in [p.get("category",""), p.get("brand",""), p.get("color","")]
            if t
        )
        st.markdown(tags, unsafe_allow_html=True)
        st.markdown(
            f"<span style='color:#10b981;font-weight:600'>${p.get('price', 0):.2f}</span>"
            f"  •  <span style='color:#94a3b8;font-size:0.85rem'>{p.get('attributes','')[:80]}</span>",
            unsafe_allow_html=True,
        )
        if show_score and "score" in p:
            pct = int(min(100, max(0, float(p["score"]) * 100))  )
            st.markdown(
                f'<div class="score-bar" style="width:{pct}%;margin-top:4px"></div>',
                unsafe_allow_html=True,
            )


# ── tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_search, tab_gen, tab_clf = st.tabs(
    ["💬 Chat", "🔍 Search", "✍️ Generate Content", "🖼️ Classify Image"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Shopping Assistant Chat")
    st.caption("Ask natural-language questions. Filters (price, color, category) are extracted automatically.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_msg = st.text_input("Your message", key="chat_input",
                                 placeholder="e.g. Show me black bags under $80")
    with col_btn:
        st.write("")
        send = st.button("Send", key="chat_send", use_container_width=True)

    if st.button("🗑️ Clear history", key="clear_history"):
        st.session_state.messages = []
        try:
            requests.delete(f"{API}/chat/history", timeout=5)
        except Exception:
            pass
        st.rerun()

    if send and user_msg.strip():
        with st.spinner("Thinking …"):
            resp = api_post("/chat", json={"query": user_msg, "k": 5})
        if resp:
            st.session_state.messages.append(
                {"role": "user", "text": user_msg}
            )
            st.session_state.messages.append(
                {"role": "bot", "text": resp.get("answer", ""), "products": resp.get("products", []),
                 "filters": resp.get("filters", {})}
            )

    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">👤 {msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-bot">🤖 {msg["text"]}</div>',
                unsafe_allow_html=True,
            )
            filters = msg.get("filters", {})
            active = {k: v for k, v in filters.items() if v}
            if active:
                st.caption(f"Detected filters: {active}")
            for p in msg.get("products", [])[:3]:
                render_product_card(p, show_score=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Search
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Semantic Product Search")
    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input("Search query", key="search_q",
                              placeholder="e.g. lightweight running shoes blue")
    with col_k:
        k = st.number_input("Top-k", min_value=1, max_value=20, value=5, key="search_k")

    if st.button("Search", key="search_btn") and query:
        with st.spinner("Searching …"):
            results = api_post("/search", json={"query": query, "k": int(k)})
        if results:
            st.success(f"Found {len(results)} results")
            for p in results:
                with st.container():
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    render_product_card(p, show_score=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Generate Content
# ─────────────────────────────────────────────────────────────────────────────
with tab_gen:
    st.subheader("AI Product Description Generator")
    st.caption("Fill in the product details and choose a writing style.")

    c1, c2 = st.columns(2)
    with c1:
        pname  = st.text_input("Product name *", key="gen_name",
                               placeholder="e.g. Nova Running Shoes - Black")
        pcat   = st.text_input("Category", key="gen_cat", placeholder="shoes")
        pbrand = st.text_input("Brand", key="gen_brand", placeholder="Nova")
    with c2:
        pcolor = st.text_input("Color", key="gen_color", placeholder="black")
        pattrs = st.text_area("Features (semicolon-separated)", key="gen_attrs",
                              placeholder="lightweight; breathable mesh; cushioned sole",
                              height=100)
        style  = st.selectbox("Writing style", ["default", "seo", "short", "luxury"], key="gen_style")

    if st.button("✨ Generate Description", key="gen_btn") and pname.strip():
        with st.spinner("Generating …"):
            resp = api_post("/generate", json={
                "name": pname, "category": pcat, "brand": pbrand,
                "color": pcolor, "attributes": pattrs, "style": style,
            })
        if resp:
            st.markdown("### Generated Description")
            st.success(resp.get("description", ""))

            if st.button("🔍 Find similar products", key="gen_similar"):
                with st.spinner("Searching …"):
                    sim = api_post("/similar", json={"product_name": pname, "k": 5})
                if sim:
                    st.markdown("#### Similar products")
                    for p in sim:
                        render_product_card(p)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Classify Image
# ─────────────────────────────────────────────────────────────────────────────
with tab_clf:
    st.subheader("Product Image Classifier")
    st.caption("Upload a product image and get the predicted category with confidence scores.")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"], key="clf_file")
    if uploaded:
        img_bytes = uploaded.read()
        col_img, col_res = st.columns([1, 2])
        with col_img:
            st.image(img_bytes, caption="Uploaded image", width=250)

        with col_res:
            with st.spinner("Classifying …"):
                resp = requests.post(
                    f"{API}/classify",
                    files={"file": (uploaded.name, img_bytes, uploaded.type)},
                    timeout=30,
                )
            if resp.status_code == 200:
                preds = resp.json().get("predictions", [])
                st.markdown("### Top-3 Predictions")
                for i, pred in enumerate(preds):
                    cat  = pred.get("category", pred.get("class", "?"))
                    conf = float(pred.get("confidence", pred.get("probability", 0)))
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else ""
                    st.markdown(f"{medal} **{cat}**  —  {conf*100:.1f}%")
                    st.progress(conf)
            elif resp.status_code == 503:
                st.warning("Classifier not available. Train it first:\n```\npython -m train.train_classifier\n```")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")

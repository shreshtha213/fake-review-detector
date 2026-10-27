"""Fake Review Detector - Streamlit front end.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from detector import Analysis, analyze
from detector.model import load_model
from ui import components as ui

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")

EXAMPLES = {
    "😬 Suspicious": (
        "AMAZING product!!! Best purchase ever, highly recommend to everyone. "
        "Absolutely perfect, five stars!!! You won't regret it! Buy it now!"
    ),
    "🙂 Genuine": (
        "I've been using this kettle for about three months. It boils 1.5 liters in roughly "
        "four minutes, which is faster than my old one. The lid is a bit stiff to open and the "
        "cable is short, but for the price it does the job well."
    ),
    "📢 Spam": (
        "Great product, works great. Use my code SAVE20 at www.dealz-shop.com for a discount!"
    ),
}
RATING_OPTIONS = ["Not given", 1, 2, 3, 4, 5]
# Strictness level -> fake-score cutoff. Stricter means a lower cutoff, so more reviews get flagged.
STRICTNESS = {"Relaxed": 0.65, "Balanced": 0.50, "Strict": 0.35}

state = st.session_state
state.setdefault("results", {})      # tab -> (Analysis, verified_badge)
state.setdefault("history", [])
state.setdefault("auto_run", False)


@st.cache_resource(show_spinner="Loading OCR engine...")
def get_ocr():
    from detector import ocr
    return ocr


@st.cache_data(show_spinner="Reading text from image...", max_entries=32)
def run_ocr(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return get_ocr().extract_review(image)


def run_analysis(tab: str, text: str, rating, verified_badge: bool = False) -> None:
    try:
        result = analyze(text, None if rating == "Not given" else int(rating))
    except ValueError as exc:
        st.warning(str(exc))
        return
    state.results[tab] = (result, verified_badge)
    state.history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": "🖼️ Screenshot" if tab == "image" else "✍️ Text",
        "Review": text[:70] + ("…" if len(text) > 70 else ""),
        "score": result.fake_probability,
    })
    st.toast("Analysis complete", icon="🔎")


def show_result(tab: str, threshold: float) -> None:
    if tab not in state.results:
        return
    analysis, verified_badge = state.results[tab]
    # Re-apply the sidebar threshold so moving the slider updates the verdict live.
    a: Analysis = dataclasses.replace(analysis, threshold=threshold)
    ui.verdict_card(a)
    ui.stat_tiles(a)
    if verified_badge:
        st.info("**Verified Purchase** badge spotted in the screenshot: a mild sign the reviewer bought it.",
                icon="🛒")

    hl, crit, mood, details = st.tabs(["🖍️ Highlights", "📊 Criteria", "💬 Sentence mood", "📋 Details"])
    with hl:
        ui.highlighted_review(a)
    with crit:
        ui.criteria_chart(a)
    with mood:
        ui.sentence_chart(a)
    with details:
        ui.criteria_list(a)


def use_example(name: str) -> None:
    state.review_text = EXAMPLES[name]
    state.auto_run = True


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎚️ Strictness")
    strictness = st.select_slider(
        "How strict should the detector be?", options=list(STRICTNESS), value="Balanced",
        help="Strict flags more reviews as fake; Relaxed flags only clear-cut cases. "
             "The verdict updates instantly.",
    )
    threshold = STRICTNESS[strictness]
    st.caption(f"A review is flagged as fake when its score is **{threshold:.0%} or more**.")
    st.markdown("### 🧠 How it works")
    st.markdown(
        "Two detectors check every review. It's flagged if **either** is confident:\n\n"
        "1. **🤖 ML model**: spots machine-generated text (TF-IDF + logistic regression).\n"
        "2. **📏 Rule checks**: hype words, stock phrases, missing details, one-sided tone, "
        "CAPS/!!!, spam links, paid-review disclosures, star rating vs. text."
    )
    bundle = load_model()
    if bundle:
        m = bundle["metrics"]
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']:.1%}")
        c2.metric("ROC AUC", f"{m['roc_auc']:.3f}")
        st.caption(f"Held-out test set · trained on {bundle['n_reviews']:,} reviews "
                   f"({bundle['dataset']}).")
    else:
        st.warning("ML model not found: using rule checks only. Run `python train_model.py`.")
    st.divider()
    st.caption("⚠️ Showcase project. Scores are probabilistic and can be wrong; don't use them as "
               "the only basis for decisions about real people or businesses.")

# ---------------------------------------------------------------- main
ui.load_css()
ui.hero()

text_tab, image_tab = st.tabs(["Shreshtha Mukherjee", "Sakshi Gholap"])

with text_tab:
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=use_example, args=(name,), width="stretch")
    text = st.text_area("Review text", key="review_text", height=150,
                        placeholder="Paste the review here...")
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    rating = c1.selectbox("Star rating (optional)", RATING_OPTIONS, key="text_rating",
                          help="Lets the detector check whether the stars match the tone of the text.")
    clicked = c2.button("🔍 Analyze review", type="primary", key="analyze_text", width="stretch")
    if clicked or state.auto_run:
        state.auto_run = False
        run_analysis("text", text, rating)
    show_result("text", threshold)

with image_tab:
    upload = st.file_uploader("Screenshot of a review", type=["png", "jpg", "jpeg", "webp"])
    if upload is None:
        state.results.pop("image", None)
    else:
        image_bytes = upload.getvalue()
        with st.expander("🖼️ Uploaded screenshot", expanded=True):
            st.image(image_bytes, width="stretch")
        ocr_result = run_ocr(image_bytes)
        if not ocr_result.text:
            st.warning("No readable text found in this image. Try a sharper or larger screenshot.")
        else:
            # Key on the file hash so a new upload refreshes the editable text.
            key = hashlib.md5(image_bytes).hexdigest()[:10]
            if state.get("image_key") != key:
                state.image_key = key
                state.results.pop("image", None)
            ocr_text = st.text_area("Extracted text (edit if the OCR got something wrong)",
                                    value=ocr_result.text, height=150, key=f"ocr_{key}")
            default = RATING_OPTIONS.index(ocr_result.rating) if ocr_result.rating else 0
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            ocr_rating = c1.selectbox("Star rating", RATING_OPTIONS, index=default, key=f"rating_{key}",
                                      help="Detected automatically when the screenshot shows 'x out of 5'.")
            if c2.button("🔍 Analyze screenshot", type="primary", key="analyze_image", width="stretch"):
                run_analysis("image", ocr_text, ocr_rating, ocr_result.verified_badge)
            show_result("image", threshold)

# ---------------------------------------------------------------- history
if state.history:
    st.markdown("### 🗂️ Your checks this session")
    df = pd.DataFrame(state.history)
    df["Verdict"] = ["🚩 Fake" if s >= threshold else "✅ Real" for s in df["score"]]
    df["Fake score"] = df["score"] * 100
    st.dataframe(
        df[["Time", "Verdict", "Fake score", "Source", "Review"]], hide_index=True, width="stretch",
        column_config={"Fake score": st.column_config.ProgressColumn(
            min_value=0, max_value=100, format="%.0f%%")},
    )
    if st.button("🧹 Clear history"):
        state.history.clear()
        st.rerun()

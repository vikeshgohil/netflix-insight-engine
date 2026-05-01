import os
import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.ml_models import build_recommendation_model, get_recommendations

st.set_page_config(
    page_title="ML Recommendations — Netflix Insight Engine",
    page_icon="🤖",
    layout="wide"
)
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

df = load_data()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Netflix Insight Engine")
    st.markdown("*Data Science + ML + AI Platform*")
    st.divider()
    st.markdown("**Recommendation settings**")

    n_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    content_filter = st.selectbox(
        "Filter recommendations by type",
        options=["All", "Movie", "TV Show"]
    )

    st.divider()
    st.markdown("**How it works**")
    st.markdown("- TF-IDF vectorizes descriptions")
    st.markdown("- Cosine Similarity finds matches")
    st.markdown("- Score closer to 1 = more similar")

# ── Page Title ────────────────────────────────────────────
st.markdown("# 🤖 ML Recommendation System")
st.markdown("Discover similar Netflix titles using TF-IDF and Cosine Similarity.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module uses two ML techniques working together:

    **Step 1 — TF-IDF Vectorization**
    Combines the description, genre and cast of every title into one text.
    TF-IDF converts this text into a matrix of numbers — 8,807 rows × 5,000 columns.
    Words that are rare and meaningful get high scores. Common words get near-zero scores.

    **Step 2 — Cosine Similarity**
    Measures the angle between any two title vectors.
    Score of 1.0 = identical. Score of 0.0 = completely different.
    The top N titles with highest scores are returned as recommendations.

    **Why these techniques?**
    Fast, interpretable, and highly effective for short text like movie descriptions.
    """)

st.divider()

# ── Build model ───────────────────────────────────────────
with st.spinner("Building recommendation model — this takes a few seconds..."):
    similarity_matrix, df_model = build_recommendation_model(df)

st.success(f"Model ready — {len(df_model):,} titles indexed")

st.divider()

# ── Section A: Title Selector ─────────────────────────────
st.markdown("### Select a title you like")

all_titles = sorted(df_model['title'].tolist())
selected_title = st.selectbox(
    "Choose a Netflix title",
    options=all_titles,
    index=all_titles.index('Squid Game') if 'Squid Game' in all_titles else 0
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    find_button = st.button(
        "Find similar titles →",
        type="primary",
        use_container_width=True
    )

with col_info:
    selected_row = df_model[df_model['title'] == selected_title]
    if not selected_row.empty:
        r = selected_row.iloc[0]
        st.markdown(
            f"**{r['type']}** · {r['listed_in']} · "
            f"Rating: {r['rating']} · Year: {r['release_year']}"
        )

# ── Section B: Results ────────────────────────────────────
if find_button:
    st.divider()
    st.markdown(f"### Because you liked: *{selected_title}*")

    with st.spinner("Finding similar titles..."):
        results = get_recommendations(
            selected_title,
            df_model,
            similarity_matrix,
            n=n_recommendations * 2
        )

    if content_filter != "All":
        results = results[results['type'] == content_filter]

    results = results.head(n_recommendations)

    if results.empty:
        st.warning("No recommendations found. Try a different title.")
    else:
        st.markdown(f"Top **{len(results)}** similar titles:")
        st.divider()

        for i, (_, row) in enumerate(results.iterrows(), 1):
            col_num, col_content, col_score = st.columns([1, 8, 2])

            with col_num:
                st.markdown(f"### {i}")

            with col_content:
                st.markdown(f"**{row['title']}**")
                st.markdown(
                    f"`{row['type']}` · {row['listed_in']} · "
                    f"Rating: {row['rating']} · Year: {row['release_year']}"
                )
                with st.expander("Read description"):
                    st.markdown(row['description'])

            with col_score:
                score = row['similarity_score']
                if score >= 0.5:
                    st.success(f"Score: {score}")
                elif score >= 0.3:
                    st.warning(f"Score: {score}")
                else:
                    st.info(f"Score: {score}")

            st.divider()

# ── Section C: Try voice input ────────────────────────────
st.markdown("### Try voice input")
st.info(
    "You can also speak a title name using the Voice Recognition module "
    "and it will automatically search for recommendations. "
    "Go to Module 11 — Voice Input to try it."
)

if st.button("Go to Voice Module →"):
    st.switch_page("pages/11_Voice_Recognition.py")

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — Sentiment Analysis →"):
        st.switch_page("pages/5_Sentiment_Analysis.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
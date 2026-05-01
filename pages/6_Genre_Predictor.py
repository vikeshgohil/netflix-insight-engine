import os
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data
from utils.ml_models import train_genre_model, predict_genre

st.set_page_config(
    page_title="Genre Predictor — Netflix Insight Engine",
    page_icon="🎯",
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
    st.markdown("**Model info**")
    st.markdown("- Algorithm: Logistic Regression")
    st.markdown("- Features: TF-IDF (5,000 terms)")
    st.markdown("- Train split: 80%")
    st.markdown("- Test split: 20%")
    st.divider()
    st.markdown("**Try these descriptions:**")
    st.markdown("*A haunted house where dark spirits terrorize a family*")
    st.markdown("*Two people fall in love during a summer vacation*")
    st.markdown("*A detective investigates a series of mysterious murders*")

# ── Page Title ────────────────────────────────────────────
st.markdown("# 🎯 Genre Predictor")
st.markdown("Type any description and ML predicts the Netflix genre.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module uses **Supervised Machine Learning** — specifically Logistic Regression:

    **Training phase** (runs once when page loads):
    1. Takes 8,807 Netflix descriptions paired with their genre labels
    2. Filters genres with at least 30 examples for reliable learning
    3. Splits data — 80% for training, 20% for testing
    4. TF-IDF converts descriptions to number matrix
    5. Logistic Regression learns word-to-genre patterns
    6. Model is tested on unseen 20% — accuracy score calculated

    **Prediction phase** (runs when you click predict):
    1. Your typed description is cleaned and vectorized
    2. The trained model checks learned word-weight patterns
    3. Returns top 3 genre predictions with confidence percentages

    **Why Logistic Regression?**
    Fast, interpretable, and highly effective for text classification 
    on datasets of this size. Accuracy of 40-65% is normal for 40+ genre classes.
    """)

st.divider()

# ── Train model ───────────────────────────────────────────
with st.spinner("Training genre prediction model..."):
    model, tfidf, le, accuracy = train_genre_model(df)

col_acc1, col_acc2, col_acc3 = st.columns(3)
with col_acc1:
    st.metric("Model accuracy", f"{accuracy}%")
with col_acc2:
    st.metric("Algorithm", "Logistic Regression")
with col_acc3:
    st.metric("Features", "5,000 TF-IDF terms")

st.divider()

# ── Section A: Prediction Input ───────────────────────────
st.markdown("### Predict genre from description")

sample_descriptions = {
    "Custom — type your own": "",
    "Horror example": "A terrifying haunted house where evil spirits torture a family trapped inside",
    "Romance example": "Two strangers fall deeply in love during a magical summer vacation in Paris",
    "Crime example": "A ruthless drug lord builds a criminal empire while evading law enforcement",
    "Documentary example": "An exploration of ancient civilizations and their remarkable achievements",
    "Comedy example": "A clumsy office worker accidentally becomes the CEO of a major corporation"
}

selected_sample = st.selectbox(
    "Choose a sample or type your own below",
    options=list(sample_descriptions.keys())
)

if selected_sample != "Custom — type your own":
    prefill_text = sample_descriptions[selected_sample]
else:
    prefill_text = ""

user_description = st.text_area(
    "Enter a movie or show description:",
    value=prefill_text,
    placeholder="Describe a movie or show in your own words...",
    height=120
)

predict_button = st.button(
    "Predict Genre →",
    type="primary",
    use_container_width=False
)

# ── Section B: Prediction Results ─────────────────────────
if predict_button:
    if user_description.strip():
        with st.spinner("Predicting genre..."):
            predictions = predict_genre(
                user_description, model, tfidf, le, top_n=3
            )

        if predictions:
            st.divider()
            st.markdown("### Prediction results")

            top_pred = predictions[0]
            confidence = top_pred['confidence']

            if confidence >= 50:
                st.success(f"Top prediction: **{top_pred['genre']}** — {confidence}% confident")
            elif confidence >= 30:
                st.warning(f"Top prediction: **{top_pred['genre']}** — {confidence}% confident")
            else:
                st.info(f"Top prediction: **{top_pred['genre']}** — {confidence}% confident")

            st.markdown("#### Top 3 predictions")
            col1, col2, col3 = st.columns(3)

            cols = [col1, col2, col3]
            colors = ["#E50914", "#ff6b6b", "#ffaaaa"]

            for i, (col, pred) in enumerate(zip(cols, predictions)):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**Rank {i + 1}**")
                        st.markdown(f"### {pred['genre']}")
                        st.metric(
                            "Confidence",
                            f"{pred['confidence']}%"
                        )

            st.divider()

            pred_df = pd.DataFrame(predictions)
            pred_df.columns = ['Genre', 'Confidence (%)']
            fig = px.bar(
                pred_df,
                x='Confidence (%)',
                y='Genre',
                orientation='h',
                title='Prediction confidence scores',
                color='Confidence (%)',
                color_continuous_scale=['#ffaaaa', '#E50914']
            )
            fig.update_layout(
                height=280,
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Could not generate prediction. Try a longer or more descriptive text.")
    else:
        st.warning("Please enter a description to predict the genre.")

st.divider()

# ── Section C: Model Performance ─────────────────────────
st.markdown("### Model performance details")

with st.expander("View model evaluation info"):
    st.markdown(f"""
    **Training results:**
    - Algorithm: Logistic Regression
    - Vectorizer: TF-IDF with 5,000 features and bigrams
    - Training set: 80% of eligible titles
    - Test set: 20% of eligible titles
    - Test accuracy: **{accuracy}%**

    **Why is accuracy not 100%?**

    Netflix titles span 40+ genres and many titles belong to multiple genres 
    simultaneously. For example a title listed as 
    "Dramas, International Movies, Romantic Movies" is assigned only 
    "Dramas" as its training label — the model never sees the other two genres 
    for that title. This natural label ambiguity limits accuracy. 
    An accuracy of 40-65% is considered very good for this type of 
    multi-class text classification problem.

    **Top-3 accuracy** (correct genre appearing in top 3 predictions) 
    is significantly higher than the single-label accuracy shown above.
    """)

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — Word Cloud →"):
        st.switch_page("pages/7_Word_Cloud.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
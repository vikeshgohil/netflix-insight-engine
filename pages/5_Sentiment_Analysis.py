import os
import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.data_loader import load_data

st.set_page_config(
    page_title="Sentiment Analysis — Netflix Insight Engine",
    page_icon="💬",
    layout="wide"
)
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

df = load_data()
analyzer = SentimentIntensityAnalyzer()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Netflix Insight Engine")
    st.markdown("*Data Science + ML + AI Platform*")
    st.divider()
    st.markdown("**Sentiment settings**")

    content_type = st.selectbox(
        "Content type",
        options=["All", "Movie", "TV Show"]
    )

    sentiment_filter = st.selectbox(
        "Filter by sentiment",
        options=["All", "Positive", "Neutral", "Negative"]
    )

    st.divider()
    st.markdown("**VADER thresholds**")
    st.markdown("- Positive: compound ≥ 0.05")
    st.markdown("- Neutral: -0.05 to 0.05")
    st.markdown("- Negative: compound ≤ -0.05")

# ── Page Title ────────────────────────────────────────────
st.markdown("# 💬 Sentiment Analysis")
st.markdown("Analyzing the emotional tone of Netflix descriptions using VADER NLP.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module uses **VADER** (Valence Aware Dictionary and Sentiment Reasoner):

    - VADER reads each Netflix description word by word
    - Every word has a pre-assigned sentiment score in VADER's dictionary
    - It considers intensifiers ("very", "extremely") and negations ("not", "never")
    - It produces a **compound score** between -1 and +1
    - Compound ≥ 0.05 → **Positive** | -0.05 to 0.05 → **Neutral** | ≤ -0.05 → **Negative**

    VADER requires no training — it uses a pre-built dictionary of 7,500 scored words.
    Perfect for short texts like Netflix descriptions.
    """)

st.divider()


# ── Run sentiment analysis ────────────────────────────────
@st.cache_data
def analyze_all_sentiments(df):
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    scores = df['description'].apply(
        lambda x: analyzer.polarity_scores(str(x))
    )
    df['sentiment_score'] = scores.apply(lambda x: round(x['compound'], 4))
    df['positive_score'] = scores.apply(lambda x: round(x['pos'], 4))
    df['negative_score'] = scores.apply(lambda x: round(x['neg'], 4))
    df['neutral_score'] = scores.apply(lambda x: round(x['neu'], 4))
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x >= 0.05
        else ('Negative' if x <= -0.05 else 'Neutral')
    )
    return df


with st.spinner("Analyzing sentiment of all 8,807 descriptions..."):
    df_sentiment = analyze_all_sentiments(df)

# ── Apply filters ─────────────────────────────────────────
filtered = df_sentiment.copy()
if content_type != "All":
    filtered = filtered[filtered['type'] == content_type]
if sentiment_filter != "All":
    filtered = filtered[filtered['sentiment'] == sentiment_filter]

# ── Section A: Metric Cards ───────────────────────────────
st.markdown("### Sentiment overview")

total = len(df_sentiment)
positive = len(df_sentiment[df_sentiment['sentiment'] == 'Positive'])
neutral = len(df_sentiment[df_sentiment['sentiment'] == 'Neutral'])
negative = len(df_sentiment[df_sentiment['sentiment'] == 'Negative'])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total analyzed", f"{total:,}")
with col2:
    st.metric("Positive", f"{positive:,}",
              delta=f"{round(positive / total * 100, 1)}%")
with col3:
    st.metric("Neutral", f"{neutral:,}",
              delta=f"{round(neutral / total * 100, 1)}%")
with col4:
    st.metric("Negative", f"{negative:,}",
              delta=f"{round(negative / total * 100, 1)}%")

st.divider()

# ── Section B: Charts ─────────────────────────────────────
st.markdown("### Sentiment distribution")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    sentiment_counts = df_sentiment['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig1 = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count',
        title='Overall sentiment distribution',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Neutral': '#95a5a6',
            'Negative': '#E50914'
        },
        hole=0.4
    )
    fig1.update_layout(height=380)
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    fig2 = px.histogram(
        df_sentiment,
        x='sentiment_score',
        nbins=50,
        title='Compound score distribution',
        color_discrete_sequence=['#E50914']
    )
    fig2.update_layout(
        height=380,
        xaxis_title='Compound score (-1 to +1)',
        yaxis_title='Number of titles'
    )
    fig2.add_vline(x=0.05, line_dash="dash",
                   line_color="green", annotation_text="Positive threshold")
    fig2.add_vline(x=-0.05, line_dash="dash",
                   line_color="red", annotation_text="Negative threshold")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Section C: Sentiment by genre ────────────────────────
st.markdown("### Sentiment by genre")

genre_sentiment = df_sentiment.copy()
genre_sentiment['genre'] = genre_sentiment['listed_in'].str.split(', ').str[0]
genre_avg = (
    genre_sentiment.groupby('genre')['sentiment_score']
    .mean()
    .round(3)
    .reset_index()
    .sort_values('sentiment_score', ascending=False)
    .head(15)
)
genre_avg.columns = ['Genre', 'Avg sentiment score']

fig3 = px.bar(
    genre_avg,
    x='Avg sentiment score',
    y='Genre',
    orientation='h',
    title='Average sentiment score by top 15 genres',
    color='Avg sentiment score',
    color_continuous_scale=['#E50914', '#f5c6c6', '#2ecc71']
)
fig3.update_layout(
    height=450,
    yaxis={'categoryorder': 'total ascending'},
    coloraxis_showscale=False
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Section D: Filtered Results Table ────────────────────
st.markdown(f"### Browsing {len(filtered):,} titles")

st.dataframe(
    filtered[[
        'title', 'type', 'listed_in',
        'sentiment', 'sentiment_score', 'description'
    ]].sort_values('sentiment_score', ascending=False),
    use_container_width=True,
    height=350
)

st.divider()

# ── Section E: Live Sentiment Tester ─────────────────────
st.markdown("### Live sentiment tester")
st.markdown("Type any sentence and see its sentiment score instantly.")

user_text = st.text_area(
    "Enter any text to analyze:",
    placeholder="e.g. This movie is absolutely brilliant and thrilling!",
    height=100
)

if st.button("Analyze sentiment →", type="primary"):
    if user_text.strip():
        scores = analyzer.polarity_scores(user_text)
        compound = scores['compound']

        if compound >= 0.05:
            label = "Positive"
            color = "success"
        elif compound <= -0.05:
            label = "Negative"
            color = "error"
        else:
            label = "Neutral"
            color = "info"

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Sentiment", label)
        with col_r2:
            st.metric("Compound score", round(compound, 4))
        with col_r3:
            st.metric("Positive score", round(scores['pos'], 4))
        with col_r4:
            st.metric("Negative score", round(scores['neg'], 4))

        if color == "success":
            st.success(f"This text is **{label}** with a compound score of {round(compound, 4)}")
        elif color == "error":
            st.error(f"This text is **{label}** with a compound score of {round(compound, 4)}")
        else:
            st.info(f"This text is **{label}** with a compound score of {round(compound, 4)}")
    else:
        st.warning("Please enter some text to analyze.")

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — Genre Predictor →"):
        st.switch_page("pages/6_Genre_Predictor.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
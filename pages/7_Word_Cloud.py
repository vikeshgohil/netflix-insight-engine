import os
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from utils.nlp_helpers import get_wordcloud_text, clean_for_nlp

st.set_page_config(
    page_title="Word Cloud — Netflix Insight Engine",
    page_icon="☁️",
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
    st.markdown("**Word cloud settings**")

    all_genres = ["All Genres"] + sorted(
        df['listed_in'].str.split(', ')
        .explode().str.strip()
        .replace('', pd.NA).dropna()
        .unique().tolist()
    )

    selected_genre = st.selectbox(
        "Select genre",
        options=all_genres
    )

    max_words = st.slider(
        "Maximum words",
        min_value=50,
        max_value=200,
        value=100,
        step=10
    )

    colormap = st.selectbox(
        "Colour scheme",
        options=["Reds", "Blues", "Greens",
                 "Oranges", "Purples", "plasma", "viridis"],
        index=0
    )

    st.divider()
    st.markdown("**How it works**")
    st.markdown("- Descriptions filtered by genre")
    st.markdown("- Stop words removed via NLTK")
    st.markdown("- Bigger word = appears more often")

# ── Page Title ────────────────────────────────────────────
st.markdown("# ☁️ Word Cloud and NLP Visualization")
st.markdown("Visual frequency analysis of words in Netflix descriptions by genre.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module performs **NLP text frequency analysis**:

    1. **Filter** — descriptions are filtered by the selected genre
    2. **Clean** — text is lowercased, punctuation and numbers removed
    3. **Stop word removal** — 179 common English words removed via NLTK
       plus custom Netflix-specific stop words (netflix, series, season etc.)
    4. **Frequency count** — remaining words counted by occurrence
    5. **Word Cloud** — words drawn at sizes proportional to frequency
    6. **Bar chart** — top 20 words shown with exact counts

    Switch genres using the sidebar dropdown to see how language 
    changes completely between Horror, Romance, Documentary etc.
    """)

st.divider()


# ── Generate word cloud ───────────────────────────────────
@st.cache_data
def generate_wordcloud_cached(genre, max_words, colormap, df):
    text = get_wordcloud_text(df, genre)
    if not text.strip():
        return None, {}

    wc = WordCloud(
        width=1200,
        height=500,
        background_color='white',
        max_words=max_words,
        colormap=colormap,
        min_font_size=10,
        collocations=False
    ).generate(text)

    word_frequencies = dict(
        sorted(wc.words_.items(),
               key=lambda x: x[1], reverse=True)[:20]
    )
    return wc, word_frequencies


with st.spinner(f"Generating word cloud for {selected_genre}..."):
    wc, word_freq = generate_wordcloud_cached(
        selected_genre, max_words, colormap, df
    )

# ── Genre info ────────────────────────────────────────────
if selected_genre == "All Genres":
    genre_count = len(df)
else:
    genre_count = len(
        df[df['listed_in'].str.contains(selected_genre, na=False)]
    )

st.markdown(f"### {selected_genre}")
st.markdown(f"Analyzing descriptions from **{genre_count:,}** titles")

st.divider()

# ── Section A: Word Cloud Image ───────────────────────────
st.markdown("### Word cloud")

if wc is None:
    st.warning("Not enough text data for this genre. Please select a different genre.")
else:
    fig_wc, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig_wc, use_container_width=True)
    plt.close()

st.divider()

# ── Section B: Top 20 Words Bar Chart ────────────────────
st.markdown("### Top 20 most frequent words")

if word_freq:
    freq_df = pd.DataFrame(
        list(word_freq.items()),
        columns=['Word', 'Relative frequency']
    ).sort_values('Relative frequency', ascending=True)

    fig_bar = px.bar(
        freq_df,
        x='Relative frequency',
        y='Word',
        orientation='h',
        title=f'Top 20 words in {selected_genre} descriptions',
        color='Relative frequency',
        color_continuous_scale=['#f5c6c6', '#E50914']
    )
    fig_bar.update_layout(
        height=550,
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        xaxis_title='Relative frequency score'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Section C: Genre comparison ──────────────────────────
st.markdown("### Compare two genres")

col_g1, col_g2 = st.columns(2)

genre_options = sorted(
    df['listed_in'].str.split(', ')
    .explode().str.strip()
    .replace('', pd.NA).dropna()
    .unique().tolist()
)

with col_g1:
    genre_a = st.selectbox(
        "Genre A",
        options=genre_options,
        index=genre_options.index('Horror Movies')
        if 'Horror Movies' in genre_options else 0
    )

with col_g2:
    genre_b = st.selectbox(
        "Genre B",
        options=genre_options,
        index=genre_options.index('Romantic Movies')
        if 'Romantic Movies' in genre_options else 1
    )

if st.button("Compare genres →", type="primary"):
    col_wc1, col_wc2 = st.columns(2)

    with col_wc1:
        with st.spinner(f"Generating {genre_a} cloud..."):
            wc_a, _ = generate_wordcloud_cached(
                genre_a, max_words, 'Reds', df
            )
        if wc_a:
            st.markdown(f"**{genre_a}**")
            fig_a, ax_a = plt.subplots(figsize=(8, 4))
            ax_a.imshow(wc_a, interpolation='bilinear')
            ax_a.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_a, use_container_width=True)
            plt.close()

    with col_wc2:
        with st.spinner(f"Generating {genre_b} cloud..."):
            wc_b, _ = generate_wordcloud_cached(
                genre_b, max_words, 'Blues', df
            )
        if wc_b:
            st.markdown(f"**{genre_b}**")
            fig_b, ax_b = plt.subplots(figsize=(8, 4))
            ax_b.imshow(wc_b, interpolation='bilinear')
            ax_b.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_b, use_container_width=True)
            plt.close()

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — Country Heatmap →"):
        st.switch_page("pages/8_Country_Heatmap.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
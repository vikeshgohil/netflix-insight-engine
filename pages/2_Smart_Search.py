import streamlit as st
import os
from utils.data_loader import load_data, filter_data, get_all_genres, get_all_countries

st.set_page_config(
    page_title="Smart Search — Netflix Insight Engine",
    page_icon="🔍",
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
    st.markdown("**Search filters**")

    content_type = st.selectbox(
        "Content type",
        options=["All", "Movie", "TV Show"]
    )

    all_genres = ["All"] + get_all_genres(df)
    genre = st.selectbox("Genre", options=all_genres)

    all_countries = ["All"] + get_all_countries(df)
    country = st.selectbox("Country", options=all_countries)

    year_range = st.slider(
        "Release year range",
        min_value=1925,
        max_value=2021,
        value=(2000, 2021)
    )

    st.divider()
    st.markdown(f"- Total titles: **{len(df):,}**")

# ── Page Title ────────────────────────────────────────────
st.markdown("# 🔍 Smart Search and Filter")
st.markdown("Search across title, cast, director and description simultaneously.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module uses the **filter_data()** function which applies up to five 
    conditions simultaneously:
    - **Keyword search** — searches across title, cast, director and description
    - **Content type** — filter by Movie or TV Show
    - **Genre** — filter by any Netflix genre category
    - **Country** — filter by country of production
    - **Year range** — filter by release year using the sidebar slider

    All filters work together in real time — results update as you type.
    """)

st.divider()

# ── Section A: Search Bar ─────────────────────────────────
st.markdown("### Search Netflix titles")

keyword = st.text_input(
    label="Search",
    placeholder="Type a title, actor name, director or keyword...",
    label_visibility="collapsed"
)

# ── Apply all filters ─────────────────────────────────────
results = filter_data(
    df,
    keyword=keyword,
    content_type=content_type,
    genre=genre,
    country=country,
    year_range=year_range
)

# ── Results count bar ─────────────────────────────────────
col_count, col_clear = st.columns([4, 1])
with col_count:
    if keyword.strip():
        st.success(f"Found **{len(results):,}** results for: *{keyword}*")
    else:
        st.info(f"Showing **{len(results):,}** titles — use filters to narrow down")

with col_clear:
    if st.button("Clear filters"):
        st.rerun()

st.divider()

# ── Section B: Results Table ──────────────────────────────
if len(results) == 0:
    st.warning("No results found. Try a different keyword or adjust your filters.")
    st.markdown("**Suggestions:**")
    st.markdown("- Check spelling of actor or director name")
    st.markdown("- Try a broader keyword like a genre word")
    st.markdown("- Widen the year range in the sidebar")
else:
    st.markdown(f"### Results — {len(results):,} titles")

    display_cols = st.multiselect(
        "Choose columns to display",
        options=['title', 'type', 'director', 'cast',
                 'country', 'release_year', 'rating',
                 'listed_in', 'description'],
        default=['title', 'type', 'listed_in',
                 'country', 'release_year', 'rating']
    )

    if display_cols:
        st.dataframe(
            results[display_cols],
            use_container_width=True,
            height=450
        )
    else:
        st.dataframe(results, use_container_width=True, height=450)

    st.caption(f"Showing all {len(results):,} matching results")

st.divider()

# ── Section C: Quick Stats on Results ────────────────────
if len(results) > 0:
    st.markdown("### Quick stats on your results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        movies_count = len(results[results['type'] == 'Movie'])
        st.metric("Movies", f"{movies_count:,}")
    with col2:
        tv_count = len(results[results['type'] == 'TV Show'])
        st.metric("TV Shows", f"{tv_count:,}")
    with col3:
        top_country = results['country'].value_counts().index[0] \
            if len(results) > 0 else "N/A"
        st.metric("Top country", top_country[:15])
    with col4:
        top_rating = results['rating'].value_counts().index[0] \
            if len(results) > 0 else "N/A"
        st.metric("Top rating", top_rating)

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — EDA Charts →"):
        st.switch_page("pages/3_EDA_Visualizations.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
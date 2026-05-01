import os
import streamlit as st
import pandas as pd
from utils.data_loader import load_data

st.set_page_config(
    page_title="Data Preview — Netflix Insight Engine",
    page_icon="📊",
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
    st.markdown("**Dataset info**")
    st.markdown(f"- Total titles: **{len(df):,}**")
    st.markdown(f"- Movies: **{len(df[df['type'] == 'Movie']):,}**")
    st.markdown(f"- TV Shows: **{len(df[df['type'] == 'TV Show']):,}**")
    st.markdown(f"- Years: **2008 — 2021**")

# ── Page Title ────────────────────────────────────────────
st.markdown("# 📊 Data Preview and EDA")
st.markdown("Explore the raw Netflix dataset — every row, every column, every stat.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module loads the Netflix CSV dataset and displays it in three ways:
    - **Metric cards** — key numbers at a glance
    - **Raw data table** — browse every title with sorting and filtering
    - **Column info** — data types and missing value counts
    - **Statistical summary** — min, max, mean for numeric columns

    All data passes through our cleaning pipeline automatically before display.
    """)

st.divider()

# ── Section A: Metric Cards ───────────────────────────────
st.markdown("### Key metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total titles", f"{len(df):,}")
with col2:
    st.metric("Total movies", f"{len(df[df['type'] == 'Movie']):,}")
with col3:
    st.metric("Total TV shows", f"{len(df[df['type'] == 'TV Show']):,}")
with col4:
    st.metric("Unique directors", f"{df['director'].replace('Unknown', pd.NA).dropna().nunique():,}")

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.metric("Year range", "2008 — 2021")
with col6:
    st.metric("Total columns", f"{len(df.columns)}")
with col7:
    missing = df.isnull().sum().sum()
    st.metric("Missing values", f"{missing}")
with col8:
    genres = df['listed_in'].str.split(', ').explode().str.strip().nunique()
    st.metric("Unique genres", f"{genres:,}")

st.divider()

# ── Section B: Raw Data Table ─────────────────────────────
st.markdown("### Raw dataset")

col_left, col_right = st.columns([1, 3])
with col_left:
    rows_to_show = st.selectbox(
        "Rows to display",
        options=[10, 25, 50, 100, 200],
        index=1
    )
with col_right:
    type_filter = st.selectbox(
        "Filter by type",
        options=["All", "Movie", "TV Show"],
        index=0
    )

if type_filter == "All":
    display_df = df.head(rows_to_show)
else:
    display_df = df[df['type'] == type_filter].head(rows_to_show)

st.dataframe(
    display_df[[
        'title', 'type', 'director', 'cast',
        'country', 'release_year', 'rating',
        'duration', 'listed_in', 'description'
    ]],
    use_container_width=True,
    height=400
)

st.caption(f"Showing {len(display_df)} of {len(df):,} total titles")

st.divider()

# ── Section C: Column Information ─────────────────────────
st.markdown("### Column information")

col_info = []
for col in df.columns:
    col_info.append({
        'Column name': col,
        'Data type': str(df[col].dtype),
        'Non-null count': int(df[col].notna().sum()),
        'Missing count': int(df[col].isna().sum()),
        'Missing %': f"{round(df[col].isna().mean() * 100, 1)}%",
        'Unique values': int(df[col].nunique())
    })

col_info_df = pd.DataFrame(col_info)
st.dataframe(col_info_df, use_container_width=True)

st.divider()

# ── Section D: Statistical Summary ────────────────────────
st.markdown("### Statistical summary")

tab1, tab2 = st.tabs(["Numeric columns", "Text columns"])

with tab1:
    numeric_cols = df[['release_year', 'duration_int', 'year_added', 'month_added']]
    st.dataframe(
        numeric_cols.describe().round(2),
        use_container_width=True
    )
    st.caption("duration_int = duration in minutes for Movies, number of seasons for TV Shows")

with tab2:
    text_summary = []
    for col in ['type', 'rating', 'listed_in', 'country', 'director']:
        top_value = df[col].value_counts().index[0] if df[col].nunique() > 0 else 'N/A'
        top_count = df[col].value_counts().iloc[0] if df[col].nunique() > 0 else 0
        text_summary.append({
            'Column': col,
            'Unique values': df[col].nunique(),
            'Most common value': top_value,
            'Most common count': top_count
        })
    st.dataframe(pd.DataFrame(text_summary), use_container_width=True)

st.divider()

# ── Next Module Button ─────────────────────────────────────
st.markdown("### Ready to explore more?")
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — Smart Search →"):
        st.switch_page("pages/2_Smart_Search.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
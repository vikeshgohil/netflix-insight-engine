import os
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(
    page_title="EDA Visualizations — Netflix Insight Engine",
    page_icon="📈",
    layout="wide"
)
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
df = load_data()

# ── Sidebar Filters ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Netflix Insight Engine")
    st.markdown("*Data Science + ML + AI Platform*")
    st.divider()
    st.markdown("**Chart filters**")

    content_type = st.selectbox(
        "Content type",
        options=["All", "Movie", "TV Show"]
    )

    year_range = st.slider(
        "Release year range",
        min_value=1925,
        max_value=2021,
        value=(2000, 2021)
    )

    st.divider()
    st.markdown(f"- Total titles: **{len(df):,}**")

# ── Apply filters ─────────────────────────────────────────
filtered = df.copy()
if content_type != "All":
    filtered = filtered[filtered['type'] == content_type]
filtered = filtered[
    filtered['release_year'].between(year_range[0], year_range[1])
]

# ── Page Title ────────────────────────────────────────────
st.markdown("# 📈 EDA Visualizations")
st.markdown("Interactive charts revealing patterns and insights in the Netflix dataset.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module performs **Exploratory Data Analysis (EDA)** — the process of 
    visually understanding data before applying machine learning.

    All six charts update together when you change the sidebar filters.
    - **Plotly Express** renders all charts — fully interactive with hover, zoom and download
    - **Pandas groupby** aggregates the data behind each chart
    - Change the content type or year range on the left to see how charts change
    """)

st.divider()
st.markdown(f"Showing **{len(filtered):,}** titles after filters")
st.divider()

# ── Chart 1 + Chart 2 ─────────────────────────────────────
st.markdown("### Content overview")
col1, col2 = st.columns(2)

with col1:
    type_counts = filtered['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    fig1 = px.pie(
        type_counts,
        names='Type',
        values='Count',
        title='Movies vs TV Shows',
        color_discrete_sequence=['#E50914', '#141414'],
        hole=0.4
    )
    fig1.update_layout(
        title_font_size=16,
        showlegend=True,
        height=380
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    top_countries = (
        filtered['country']
        .str.split(', ')
        .explode()
        .str.strip()
        .replace('Unknown', pd.NA)
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_countries.columns = ['Country', 'Count']
    fig2 = px.bar(
        top_countries,
        x='Count',
        y='Country',
        orientation='h',
        title='Top 10 content producing countries',
        color='Count',
        color_continuous_scale=['#f5c6c6', '#E50914']
    )
    fig2.update_layout(
        title_font_size=16,
        height=380,
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Chart 3: Content added per year ───────────────────────
st.markdown("### Netflix growth over time")

yearly = (
    filtered.dropna(subset=['year_added'])
    .groupby(['year_added', 'type'])
    .size()
    .reset_index(name='count')
)
yearly['year_added'] = yearly['year_added'].astype(int)

fig3 = px.line(
    yearly,
    x='year_added',
    y='count',
    color='type',
    title='Titles added to Netflix per year',
    markers=True,
    color_discrete_map={'Movie': '#E50914', 'TV Show': '#141414'}
)
fig3.update_layout(
    title_font_size=16,
    height=380,
    xaxis_title='Year',
    yaxis_title='Number of titles added',
    legend_title='Content type'
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Chart 4 + Chart 5 ─────────────────────────────────────
st.markdown("### Genre and rating breakdown")
col3, col4 = st.columns(2)

with col3:
    genres = (
        filtered['listed_in']
        .str.split(', ')
        .explode()
        .str.strip()
        .value_counts()
        .head(12)
        .reset_index()
    )
    genres.columns = ['Genre', 'Count']
    fig4 = px.bar(
        genres,
        x='Count',
        y='Genre',
        orientation='h',
        title='Top 12 genres',
        color='Count',
        color_continuous_scale=['#f5c6c6', '#E50914']
    )
    fig4.update_layout(
        title_font_size=16,
        height=420,
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    ratings = (
        filtered['rating']
        .replace('Unknown', pd.NA)
        .dropna()
        .value_counts()
        .reset_index()
    )
    ratings.columns = ['Rating', 'Count']
    fig5 = px.bar(
        ratings,
        x='Rating',
        y='Count',
        title='Content by age rating',
        color='Count',
        color_continuous_scale=['#f5c6c6', '#E50914']
    )
    fig5.update_layout(
        title_font_size=16,
        height=420,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── Chart 6: Duration distribution ───────────────────────
st.markdown("### Duration distribution")

tab1, tab2 = st.tabs(["Movie duration (minutes)", "TV Show seasons"])

with tab1:
    movies_dur = filtered[
        (filtered['type'] == 'Movie') &
        (filtered['duration_int'].notna())
        ]
    if len(movies_dur) > 0:
        fig6 = px.histogram(
            movies_dur,
            x='duration_int',
            nbins=40,
            title='Movie duration distribution',
            color_discrete_sequence=['#E50914']
        )
        fig6.update_layout(
            title_font_size=16,
            height=360,
            xaxis_title='Duration (minutes)',
            yaxis_title='Number of movies'
        )
        st.plotly_chart(fig6, use_container_width=True)
        avg_dur = round(movies_dur['duration_int'].mean(), 1)
        st.info(f"Average movie duration: **{avg_dur} minutes**")
    else:
        st.info("No movie duration data available for selected filters.")

with tab2:
    shows_dur = filtered[
        (filtered['type'] == 'TV Show') &
        (filtered['duration_int'].notna())
        ]
    if len(shows_dur) > 0:
        fig7 = px.histogram(
            shows_dur,
            x='duration_int',
            nbins=15,
            title='TV Show seasons distribution',
            color_discrete_sequence=['#141414']
        )
        fig7.update_layout(
            title_font_size=16,
            height=360,
            xaxis_title='Number of seasons',
            yaxis_title='Number of TV shows'
        )
        st.plotly_chart(fig7, use_container_width=True)
        avg_seasons = round(shows_dur['duration_int'].mean(), 1)
        st.info(f"Average number of seasons: **{avg_seasons}**")
    else:
        st.info("No TV show duration data available for selected filters.")

st.divider()

# ── Chart 7: Top directors ────────────────────────────────
st.markdown("### Top directors")

top_directors = (
    filtered[filtered['director'] != 'Unknown']['director']
    .value_counts()
    .head(10)
    .reset_index()
)
top_directors.columns = ['Director', 'Titles']

fig8 = px.bar(
    top_directors,
    x='Titles',
    y='Director',
    orientation='h',
    title='Top 10 directors by number of titles',
    color='Titles',
    color_continuous_scale=['#f5c6c6', '#E50914']
)
fig8.update_layout(
    title_font_size=16,
    height=400,
    yaxis={'categoryorder': 'total ascending'},
    coloraxis_showscale=False
)
st.plotly_chart(fig8, use_container_width=True)

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — ML Recommendations →"):
        st.switch_page("pages/4_Recommendations.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
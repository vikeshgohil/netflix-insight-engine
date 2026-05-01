import os
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(
    page_title="Country Heatmap — Netflix Insight Engine",
    page_icon="🌍",
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
    st.markdown("**Map settings**")

    content_type = st.radio(
        "Content type",
        options=["All", "Movie", "TV Show"]
    )

    top_n = st.slider(
        "Top N countries in table",
        min_value=5,
        max_value=30,
        value=15
    )

    st.divider()
    st.markdown("**How it works**")
    st.markdown("- Multi-country cells are split")
    st.markdown("- Each country counted separately")
    st.markdown("- Darker = more titles")

# ── Page Title ────────────────────────────────────────────
st.markdown("# 🌍 Country Heatmap")
st.markdown("World map showing Netflix content production by country.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module builds a **Choropleth world map** using Plotly:

    1. **Split multi-country values** — a cell like "United States, India" 
       is split into two separate entries, one per country
    2. **Count titles per country** — each country gets its own title count
    3. **Plotly Choropleth** — maps country names to world geography automatically
    4. **Colour scale** — light colour = few titles, dark red = many titles
    5. **Hover tooltip** — shows exact country name and title count

    A choropleth map is a thematic map where regions are coloured 
    based on a data value — the same technique used in election maps 
    and weather maps.
    """)

st.divider()


# ── Build country data ────────────────────────────────────
@st.cache_data
def build_country_data(df, content_type):
    filtered = df.copy()
    if content_type != "All":
        filtered = filtered[filtered['type'] == content_type]

    countries = (
        filtered['country']
        .str.split(', ')
        .explode()
        .str.strip()
        .replace('Unknown', pd.NA)
        .dropna()
    )
    country_counts = (
        countries
        .value_counts()
        .reset_index()
    )
    country_counts.columns = ['Country', 'Titles']
    return country_counts


with st.spinner("Building country data..."):
    country_data = build_country_data(df, content_type)

# ── Metric Cards ──────────────────────────────────────────
st.markdown("### Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Countries represented", f"{len(country_data):,}")
with col2:
    top_country = country_data.iloc[0]['Country']
    top_count = country_data.iloc[0]['Titles']
    st.metric("Top country", top_country)
with col3:
    st.metric("Top country titles", f"{top_count:,}")
with col4:
    if content_type == "All":
        total = len(df)
    elif content_type == "Movie":
        total = len(df[df['type'] == 'Movie'])
    else:
        total = len(df[df['type'] == 'TV Show'])
    st.metric("Total titles", f"{total:,}")

st.divider()

# ── Section A: World Map ──────────────────────────────────
st.markdown("### World map — titles by country")

fig_map = px.choropleth(
    country_data,
    locations='Country',
    locationmode='country names',
    color='Titles',
    title=f'Netflix content by country — {content_type}',
    color_continuous_scale=[
        '#fff5f5', '#ffcccc', '#ff9999',
        '#ff4444', '#E50914', '#8b0000'
    ],
    hover_name='Country',
    hover_data={'Titles': True}
)

fig_map.update_layout(
    height=520,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='lightgray',
        showland=True,
        landcolor='#f8f8f8',
        showocean=True,
        oceancolor='#e8f4fd',
        showlakes=False,
        projection_type='natural earth'
    ),
    coloraxis_colorbar=dict(
        title="Titles",
        thickness=15,
        len=0.7
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig_map, use_container_width=True)
st.caption("Hover over any country to see the exact title count. Scroll to zoom.")

st.divider()

# ── Section B: Top Countries Table + Bar Chart ────────────
st.markdown(f"### Top {top_n} countries")

col_table, col_bar = st.columns([1, 2])

with col_table:
    top_countries_display = country_data.head(top_n).copy()
    top_countries_display.index = range(1, len(top_countries_display) + 1)
    top_countries_display.index.name = "Rank"
    st.dataframe(
        top_countries_display,
        use_container_width=True,
        height=420
    )

with col_bar:
    fig_bar = px.bar(
        country_data.head(top_n),
        x='Titles',
        y='Country',
        orientation='h',
        title=f'Top {top_n} countries by number of titles',
        color='Titles',
        color_continuous_scale=['#f5c6c6', '#E50914']
    )
    fig_bar.update_layout(
        height=420,
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Section C: Content type comparison ───────────────────
st.markdown("### Movies vs TV Shows by country")
st.markdown("See whether a country produces more Movies or TV Shows on Netflix.")

compare_n = st.slider("Number of countries to compare", 5, 20, 10)

movies_data = build_country_data(df, "Movie").head(compare_n)
movies_data['Type'] = 'Movie'

tvshows_data = build_country_data(df, "TV Show").head(compare_n)
tvshows_data['Type'] = 'TV Show'

top_countries_list = country_data.head(compare_n)['Country'].tolist()

movies_filtered = movies_data[
    movies_data['Country'].isin(top_countries_list)
]
tvshows_filtered = tvshows_data[
    tvshows_data['Country'].isin(top_countries_list)
]

comparison_df = pd.concat([movies_filtered, tvshows_filtered])

if len(comparison_df) > 0:
    fig_compare = px.bar(
        comparison_df,
        x='Titles',
        y='Country',
        color='Type',
        orientation='h',
        barmode='group',
        title=f'Movies vs TV Shows — top {compare_n} countries',
        color_discrete_map={
            'Movie': '#E50914',
            'TV Show': '#141414'
        }
    )
    fig_compare.update_layout(
        height=450,
        yaxis={'categoryorder': 'total ascending'},
        legend_title='Content type'
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — Content Timeline →"):
        st.switch_page("pages/9_Content_Timeline.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
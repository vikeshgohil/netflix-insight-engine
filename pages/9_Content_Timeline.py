import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data

st.set_page_config(
    page_title="Content Timeline — Netflix Insight Engine",
    page_icon="📅",
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
    st.markdown("**Timeline settings**")

    content_type = st.selectbox(
        "Content type",
        options=["All", "Movie", "TV Show"]
    )

    st.divider()
    st.markdown("**Key insight**")
    st.markdown("- Netflix started in 2008")
    st.markdown("- Peak growth: 2019")
    st.markdown("- 1,960 titles added in 2019")
    st.markdown("- Shift to TV Shows after 2018")

# ── Page Title ────────────────────────────────────────────
st.markdown("# 📅 Content Growth Timeline")
st.markdown("How Netflix grew from a small library to a global platform — 2008 to 2021.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module tells the **data story of Netflix's growth**:

    - **year_added** and **month_added** columns (created during data cleaning) 
      are grouped to count titles added each year and month
    - **Animated bar chart** — Plotly's animation_frame parameter creates 
      one frame per year automatically — no complex animation code needed
    - **Monthly heatmap** — a pivot table of year × month counts 
      reveals seasonal patterns in content addition
    - **Dual line chart** — Movies vs TV Shows growth comparison 
      shows when Netflix shifted its content strategy

    The 2019 spike reflects Netflix's heavy investment in international 
    original content — Indian, Korean, Spanish and German productions 
    all expanded significantly that year.
    """)

st.divider()


# ── Build timeline data ───────────────────────────────────
@st.cache_data
def build_timeline_data(df, content_type):
    filtered = df.dropna(subset=['year_added']).copy()
    filtered['year_added'] = filtered['year_added'].astype(int)
    filtered['month_added'] = filtered['month_added'].astype('Int64')

    if content_type != "All":
        filtered = filtered[filtered['type'] == content_type]

    yearly = (
        filtered.groupby(['year_added', 'type'])
        .size()
        .reset_index(name='count')
    )

    yearly_total = (
        filtered.groupby('year_added')
        .size()
        .reset_index(name='count')
    )

    monthly_pivot = (
        filtered.dropna(subset=['month_added'])
        .groupby(['year_added', 'month_added'])
        .size()
        .reset_index(name='count')
        .pivot(index='year_added', columns='month_added', values='count')
        .fillna(0)
        .astype(int)
    )

    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_pivot.columns = [
        month_names.get(c, c) for c in monthly_pivot.columns
    ]

    return filtered, yearly, yearly_total, monthly_pivot


with st.spinner("Building timeline data..."):
    filtered_df, yearly, yearly_total, monthly_pivot = build_timeline_data(
        df, content_type
    )

# ── Section A: Metric Cards ───────────────────────────────
st.markdown("### Key milestones")

peak_year = int(yearly_total.loc[yearly_total['count'].idxmax(), 'year_added'])
peak_count = int(yearly_total['count'].max())
first_year = int(yearly_total['year_added'].min())
total_years = int(yearly_total['year_added'].max()) - first_year

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("First title added", str(first_year))
with col2:
    st.metric("Peak growth year", str(peak_year))
with col3:
    st.metric("Titles in peak year", f"{peak_count:,}")
with col4:
    st.metric("Years of growth", f"{total_years} years")

st.divider()

# ── Section B: Animated Bar Chart ────────────────────────
st.markdown("### Animated growth chart")
st.markdown("Watch Netflix's library grow year by year.")

yearly_anim = (
    df.dropna(subset=['year_added'])
    .copy()
)
yearly_anim['year_added'] = yearly_anim['year_added'].astype(int)

cumulative = (
    yearly_anim.groupby(['year_added', 'type'])
    .size()
    .reset_index(name='count')
    .sort_values('year_added')
)

cumulative['cumulative'] = cumulative.groupby('type')['count'].cumsum()

fig_anim = px.bar(
    cumulative,
    x='type',
    y='count',
    color='type',
    animation_frame='year_added',
    title='Titles added per year — animated',
    color_discrete_map={
        'Movie': '#E50914',
        'TV Show': '#F5A623'
    },
    range_y=[0, cumulative['count'].max() * 1.2],
    labels={'count': 'Titles added', 'type': 'Content type'}
)
fig_anim.update_layout(
    height=420,
    showlegend=True,
    xaxis_title='Content type',
    yaxis_title='Number of titles added'
)
fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
fig_anim.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 400

st.plotly_chart(fig_anim, use_container_width=True)
st.caption("Click the play button to watch the animation. Use the slider to jump to any year.")

st.divider()

# ── Section C: Yearly Line Chart ─────────────────────────
st.markdown("### Year by year growth")

yearly_all = (
    df.dropna(subset=['year_added'])
    .copy()
)
yearly_all['year_added'] = yearly_all['year_added'].astype(int)
yearly_line = (
    yearly_all.groupby(['year_added', 'type'])
    .size()
    .reset_index(name='count')
)

fig_line = px.line(
    yearly_line,
    x='year_added',
    y='count',
    color='type',
    title='Titles added per year — Movies vs TV Shows',
    markers=True,
    color_discrete_map={
        'Movie': '#E50914',
        'TV Show': '#141414'
    }
)
fig_line.update_layout(
    height=380,
    xaxis_title='Year',
    yaxis_title='Number of titles added',
    legend_title='Content type'
)
fig_line.add_vline(
    x=2019,
    line_dash="dash",
    line_color="gray",
    annotation_text="Peak 2019",
    annotation_position="top right"
)
st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ── Section D: Monthly Heatmap ────────────────────────────
st.markdown("### Monthly content addition heatmap")
st.markdown("Which months does Netflix add the most content?")

if len(monthly_pivot) > 0:
    fig_heat = px.imshow(
        monthly_pivot,
        title='Content additions by year and month',
        color_continuous_scale=[
            '#fff5f5', '#ffcccc',
            '#ff6666', '#E50914', '#8b0000'
        ],
        aspect='auto',
        labels=dict(
            x='Month',
            y='Year',
            color='Titles added'
        )
    )
    fig_heat.update_layout(
        height=400,
        xaxis_title='Month',
        yaxis_title='Year'
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Darker cells = more titles added that month. "
        "Notice the December and January spikes — Netflix adds "
        "heavily before and after the holiday season."
    )

st.divider()

# ── Section E: Cumulative growth ─────────────────────────
st.markdown("### Cumulative library growth")

cumulative_total = (
    df.dropna(subset=['year_added'])
    .copy()
)
cumulative_total['year_added'] = cumulative_total['year_added'].astype(int)
cum_by_year = (
    cumulative_total.groupby(['year_added', 'type'])
    .size()
    .reset_index(name='count')
    .sort_values('year_added')
)
cum_by_year['cumulative'] = cum_by_year.groupby('type')['count'].cumsum()

fig_cum = px.area(
    cum_by_year,
    x='year_added',
    y='cumulative',
    color='type',
    title='Cumulative Netflix library size over time',
    color_discrete_map={
        'Movie': '#E50914',
        'TV Show': '#141414'
    },
    labels={
        'cumulative': 'Total titles',
        'year_added': 'Year'
    }
)
fig_cum.update_layout(
    height=380,
    xaxis_title='Year',
    yaxis_title='Total titles in library',
    legend_title='Content type'
)
st.plotly_chart(fig_cum, use_container_width=True)

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — AI Chatbot →"):
        st.switch_page("pages/10_AI_Chatbot.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
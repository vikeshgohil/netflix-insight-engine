import streamlit as st
import os
from utils.data_loader import load_data

st.set_page_config(
    page_title="Netflix Insight Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
#css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
#with open(css_path) as f:
 #   st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# ===== CSS =====
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #141414, #E50914 );
    /*background-color: #10131; */
    color: #ffffff;
}

/* HERO SECTION */
.hero {
    background: linear-gradient(135deg, #141414, #000000);
    padding: 35px;
    border-radius: 14px;
    margin-bottom: 25px;
    border: 1px solid #222;
}

/* HERO TEXT */
.hero h1 {
    color: #ffffff;
    font-size: 38px;
    font-weight: 700;
}

.hero p {
    color: #bbbbbb;
    font-size: 15px;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background: #1c1c1c;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #333;
}

/* FEATURE CARDS */
.card {
    background: #1a1a1a;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #2a2a2a;
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-6px);
    border: 1px solid #f5d2cf;
    box-shadow: 0 0 20px rgba(229,9,20,0.2);
}

/* CARD TEXT */
.card h4 {
    color: #ffffff;
    margin-bottom: 6px;
}

.card p {
    color: #aaaaaa;
    font-size: 13px;
}

/* BADGES */
.badge {
    display: inline-block;
    background: #2a2a2a;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    color: #ddd;
    margin-right: 5px;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #E50914, #b20710);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px;
    font-weight: 500;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ff1e2d, #c40812);
    transform: scale(1.03);
}

/* SIDEBAR (SAFE) */
section[data-testid="stSidebar"] {
    background-color: #030101; 

}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #777;
    font-size: 12px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
df = load_data()

# ===== SIDEBAR =====
with st.sidebar:
    st.title("🎬 Netflix Insight Engine")
    st.markdown("### Data Science + ML + AI")

    st.divider()
    st.subheader("Dataset Info")
    st.write(f"Total Titles: {len(df):,}")
    st.write(f"Movies: {len(df[df['type']=='Movie']):,}")
    st.write(f"TV Shows: {len(df[df['type']=='TV Show']):,}")

    st.divider()
    st.subheader("Project Info")
    st.write("MCA Final Year Project")
    st.write("Built with Python + Streamlit")

# ===== HERO =====
st.markdown("""
<div class="hero">
    <h1>Netflix Insight Engine</h1>
    <p>
        Explore Netflix data using EDA, ML recommendations, NLP sentiment analysis,
        AI chatbot and voice recognition — all in one intelligent platform.
    </p>
</div>
""", unsafe_allow_html=True)

# ===== METRICS =====
st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Titles", len(df))
c2.metric("Movies", len(df[df['type']=='Movie']))
c3.metric("TV Shows", len(df[df['type']=='TV Show']))

countries = df['country'].str.split(', ').explode().str.strip()
countries = countries[countries != 'Unknown'].nunique()
c4.metric("Countries", countries)

st.divider()

# ===== FEATURE CARD FUNCTION =====
def card(title, desc, modules, btn, page, key):
    st.markdown(f"""
    <div class="card">
        <h4>{title}</h4>
        <p>{desc}</p>
        <div>
            {"".join([f'<span class="badge">{m}</span>' for m in modules])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(btn, key=key):
        st.switch_page(page)

# ===== MODULES =====
st.subheader("🚀 Explore Modules")

r1c1, r1c2, r1c3 = st.columns(3)

with r1c1:
    card("📊 EDA & Data", "Explore dataset visually",
         ["M1","M2","M3"], "Open", "pages/1_Data_Preview.py", "eda")

with r1c2:
    card("🤖 ML Intelligence", "Recommendation system",
         ["M4","M6"], "Open", "pages/4_Recommendations.py", "ml")

with r1c3:
    card("💬 NLP", "Sentiment analysis",
         ["M5","M7"], "Open", "pages/5_Sentiment_Analysis.py", "nlp")

r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    card("🧠 AI Chatbot", "Ask anything using AI",
         ["M10"], "Open", "pages/10_AI_Chatbot.py", "chat")

with r2c2:
    card("🎙 Voice Input", "Voice search",
         ["M11"], "Open", "pages/11_Voice_Recognition.py", "voice")

with r2c3:
    card("🌍 Advanced Viz", "Charts & insights",
         ["M8","M9"], "Open", "pages/8_Country_Heatmap.py", "viz")

# ===== FOOTER =====
st.markdown("""
<div class="footer">
Built with Streamlit · Netflix Insight Engine · MCA Project
</div>
""", unsafe_allow_html=True)
import os
import streamlit as st
import google.generativeai as genai
from utils.data_loader import load_data

st.set_page_config(
    page_title="AI Chatbot — Netflix Insight Engine",
    page_icon="🧠",
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
    st.markdown("**Chatbot info**")
    st.markdown("- Powered by Google Gemini")
    st.markdown("- Free tier — no credit card")
    st.markdown("- Context-aware conversation")
    st.markdown("- Netflix expert assistant")
    st.divider()
    st.markdown("**Get free API key:**")
    st.markdown("aistudio.google.com")
    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Page Title ────────────────────────────────────────────
st.markdown("# 🧠 AI Chatbot")
st.markdown("Your Netflix expert assistant — powered by Google Gemini AI.")

with st.expander("ℹ️ How does this module work?"):
    st.markdown("""
    This module connects to **Google Gemini AI** — Google's large language model:

    1. **System prompt** — tells Gemini it is a Netflix expert assistant
    2. **User question** — your message is sent to the Gemini API
    3. **Context memory** — full conversation history sent with every message
       so Gemini remembers what was discussed earlier
    4. **st.session_state** — Streamlit's built-in memory stores the 
       chat history across page interactions
    5. **Free tier** — 60 requests per minute, no credit card required

    **Get your free API key:**
    Go to aistudio.google.com → Sign in with Google → Create API Key → Copy it
    """)

st.divider()

# ── API Key Input ─────────────────────────────────────────
st.markdown("### Setup")

api_key = st.text_input(
    "Enter your Google Gemini API key:",
    type="password",
    placeholder="AIza...",
    help="Get your free key at aistudio.google.com/app/apikey"
)

if not api_key.strip():
    st.info(
        "Please enter your Gemini API key above to start chatting. "
        "Get a free key at aistudio.google.com/app/apikey — "
        "no credit card required."
    )
    st.stop()

# ── Configure Gemini ──────────────────────────────────────
try:
    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Could not configure Gemini API: {str(e)}")
    st.stop()

# ── System prompt ─────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a Netflix content expert assistant built into the 
Netflix Insight Engine — a data science project analyzing Netflix's library 
of {len(df):,} titles from 2008 to 2021.

You have deep knowledge about:
- Netflix movies and TV shows — genres, directors, actors, plots
- Data science concepts — EDA, machine learning, NLP, sentiment analysis
- The Netflix dataset — 8,807 titles, 12 columns, countries, ratings, genres
- Recommendations — suggest titles based on user preferences
- Analytics insights — trends, popular genres, top countries

Be friendly, specific and helpful. When recommending content always mention 
the genre and a brief reason why it matches. Keep responses concise but 
informative. If asked about data science topics related to this project, 
explain them clearly."""

# ── Initialize session state ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggested questions ───────────────────────────────────
st.divider()
st.markdown("### Suggested questions")

suggestion_cols = st.columns(4)
suggestions = [
    "What are the best Korean thrillers on Netflix?",
    "Which director has the most Netflix titles?",
    "Recommend a good documentary to watch",
    "What genres are most common on Netflix?"
]

for i, (col, suggestion) in enumerate(zip(suggestion_cols, suggestions)):
    with col:
        if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": suggestion
            })
            with st.spinner("Thinking..."):
                try:
                    history_text = "\n".join([
                        f"{m['role'].upper()}: {m['content']}"
                        for m in st.session_state.messages[:-1]
                    ])
                    full_prompt = f"{SYSTEM_PROMPT}\n\nConversation so far:\n{history_text}\n\nUSER: {suggestion}"
                    response = model.generate_content(full_prompt)
                    assistant_reply = response.text
                except Exception as e:
                    assistant_reply = f"Sorry I encountered an error: {str(e)}"

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_reply
            })
            st.rerun()

st.divider()

# ── Chat Interface ────────────────────────────────────────
st.markdown("### Conversation")

if len(st.session_state.messages) == 0:
    st.markdown(
        "👋 Hello! I am your Netflix expert assistant. "
        "Ask me anything about Netflix shows, movies, genres, "
        "recommendations or data science concepts used in this project!"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ────────────────────────────────────────────
if prompt := st.chat_input(
        "Ask about Netflix shows, movies, genres, recommendations..."
):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history_text = "\n".join([
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.messages[:-1]
                ])
                full_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Conversation so far:\n{history_text}\n\n"
                    f"USER: {prompt}\n\n"
                    f"Please respond as the Netflix expert assistant."
                )
                response = model.generate_content(full_prompt)
                assistant_reply = response.text

            except Exception as e:
                error_msg = str(e)
                if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
                    assistant_reply = (
                        "Your API key appears to be invalid. "
                        "Please check it and try again. "
                        "Get a free key at aistudio.google.com/app/apikey"
                    )
                elif "quota" in error_msg.lower():
                    assistant_reply = (
                        "API quota exceeded. "
                        "Please wait a minute and try again. "
                        "The free tier allows 60 requests per minute."
                    )
                else:
                    assistant_reply = f"Sorry I encountered an error: {error_msg}"

        st.markdown(assistant_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

# ── Stats ─────────────────────────────────────────────────
if len(st.session_state.messages) > 0:
    st.divider()
    col_stat1, col_stat2, col_nav = st.columns([1, 1, 4])
    with col_stat1:
        st.metric(
            "Messages exchanged",
            len(st.session_state.messages)
        )
    with col_stat2:
        st.metric(
            "Your questions",
            len([m for m in st.session_state.messages
                 if m['role'] == 'user'])
        )

# ── Next Module Button ────────────────────────────────────
st.divider()
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
with col_nav1:
    if st.button("Next — Voice Recognition →"):
        st.switch_page("pages/11_Voice_Recognition.py")
with col_nav2:
    if st.button("← Back to Home"):
        st.switch_page("app.py")